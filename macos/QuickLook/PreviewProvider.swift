// Quick Look preview of MR spectroscopy files. The Python bundled in this extension runs inside
// the extension's own process (a sandboxed extension may not start other programs): libpython is
// loaded once, MIOX writes the preview page (HTML) into the extension's temporary folder, and the
// interpreter stays loaded, so the next files preview without starting Python again. A file MIOX
// cannot preview makes the request fail, so Finder shows its usual icon view.

import Foundation
import OSLog
import QuickLookUI
import UniformTypeIdentifiers

enum PreviewError: Error {
    case pythonMissing(String)
    case failed(String)
}

// The embedded interpreter: Python's C API reached through dlopen/dlsym.
final class EmbeddedPython {
    private typealias RunString = @convention(c) (UnsafePointer<CChar>) -> Int32
    private typealias GILEnsure = @convention(c) () -> Int32
    private typealias GILRelease = @convention(c) (Int32) -> Void

    private let runString: RunString
    private let ensure: GILEnsure
    private let release: GILRelease

    init(home: URL, caches: URL) throws {
        let library = home.appendingPathComponent("lib/libpython3.12.dylib").path
        guard let handle = dlopen(library, RTLD_NOW | RTLD_GLOBAL) else {
            throw PreviewError.pythonMissing(String(cString: dlerror()))
        }
        func symbol<T>(_ name: String, _ type: T.Type) throws -> T {
            guard let pointer = dlsym(handle, name) else { throw PreviewError.pythonMissing(name) }
            return unsafeBitCast(pointer, to: type)
        }
        setenv("PYTHONHOME", home.path, 1)
        setenv("PYTHONNOUSERSITE", "1", 1)
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1)
        setenv("MPLCONFIGDIR", caches.appendingPathComponent("matplotlib").path, 1)
        setenv("MPLBACKEND", "Agg", 1)
        let initialize = try symbol("Py_InitializeEx", (@convention(c) (Int32) -> Void).self)
        let saveThread = try symbol("PyEval_SaveThread", (@convention(c) () -> OpaquePointer?).self)
        runString = try symbol("PyRun_SimpleString", RunString.self)
        ensure = try symbol("PyGILState_Ensure", GILEnsure.self)
        release = try symbol("PyGILState_Release", GILRelease.self)
        initialize(0)  // no signal handlers: they belong to the host
        _ = saveThread()  // let any thread take the interpreter lock
    }

    func run(_ code: String) -> Int32 {
        let state = ensure()
        defer { release(state) }
        return runString(code)
    }
}

final class PreviewProvider: QLPreviewProvider, QLPreviewingController {
    private static let log = Logger(subsystem: "io.github.julianmer.miox", category: "preview")
    private static let lock = NSLock()
    private static var python: EmbeddedPython?

    func providePreview(for request: QLFilePreviewRequest) async throws -> QLPreviewReply {
        let file = request.fileURL
        Self.log.info("preview of \(file.lastPathComponent, privacy: .public)")
        let page: Data
        do {
            page = try Self.render(file)
        } catch {
            Self.log.error("no preview: \(String(describing: error), privacy: .public)")
            throw error
        }
        return QLPreviewReply(dataOfContentType: .html, contentSize: CGSize(width: 960, height: 720)) { _ in
            page
        }
    }

    private static func render(_ file: URL) throws -> Data {
        lock.lock()
        defer { lock.unlock() }
        if python == nil {
            guard let resources = Bundle.main.resourceURL else {
                throw PreviewError.pythonMissing("no resources")
            }
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            python = try EmbeddedPython(home: resources.appendingPathComponent("python"),
                                        caches: caches)
        }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("html")
        defer {
            for suffix in ["", ".status", ".log"] {
                try? FileManager.default.removeItem(atPath: out.path + suffix)
            }
        }
        let code = """
            import sys
            log = open(\(literal(out.path + ".log")), "w")
            sys.stderr = log
            try:
                import miox.cli
                status = miox.cli.main(["preview", \(literal(file.path)), "--to", \(literal(out.path))])
            except BaseException:
                import traceback
                traceback.print_exc(file=log)
                status = 1
            sys.stderr = sys.__stderr__
            log.close()
            open(\(literal(out.path + ".status")), "w").write(str(status))
            """
        _ = python!.run(code)
        let status = (try? String(contentsOfFile: out.path + ".status", encoding: .utf8)) ?? "?"
        guard status == "0", let data = try? Data(contentsOf: out) else {
            let message = (try? String(contentsOfFile: out.path + ".log", encoding: .utf8)) ?? ""
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            try? "\(file.path)\n\(message)".write(to: caches.appendingPathComponent("last-error.log"),
                                                    atomically: true, encoding: .utf8)
            throw PreviewError.failed("status \(status): \(message)")
        }
        return data
    }

    // a Python string literal: JSON strings are valid Python (slashes left unescaped)
    private static func literal(_ text: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: [text], options: .withoutEscapingSlashes)
        return String(String(data: data, encoding: .utf8)!.dropFirst().dropLast())
    }
}
