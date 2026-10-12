// The app that carries the MIOX Quick Look extension. Opening it once registers the extension;
// it then says where to switch it on and quits.

import AppKit

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let alert = NSAlert()
alert.messageText = "MIOX Quick Look is installed"
alert.informativeText = """
    Select an MR spectroscopy file in Finder and press the space bar.

    If no spectrum appears, switch the extension on in System Settings → General → \
    Login Items & Extensions → Quick Look → MIOX Quick Look.
    """
alert.addButton(withTitle: "OK")
app.activate(ignoringOtherApps: true)
alert.runModal()
