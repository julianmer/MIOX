#!/bin/bash
# Builds MIOX.app: a small host app carrying the Quick Look extension, which holds a standalone
# Python with MIOX installed and runs it inside its own process (a sandboxed extension may not
# start other programs). Signed for this Mac only (ad hoc). Usage: macos/build.sh [--quick] [--install]
#   --quick    keep the bundled Python of the last build (Swift and Info.plist changes only)
#   --install  copy the app to ~/Applications, register the extension and reset Quick Look.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/macos"
BUILD="$SRC/build"
APP="$BUILD/MIOX.app"
APPEX="$APP/Contents/PlugIns/MIOXQuickLook.appex"
PYTHON_VERSION="3.12"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/miox/__init__.py")"
TARGET="arm64-apple-macos12.0"

# file extensions claimed for MR data that macOS has no type for, and the system's own types;
# public.data catches files without an extension (Philips XX_0001, Bruker fid/2dseq/ser), which
# MIOX declines at once when they are not MR data
EXTENSIONS=(dat 7 rda sdat spar ima basis coord table print mrui dpt ddf cmplx rsd rsp mrd data
            list mat jdx dx jdf tnt ft ft2 ucsf nmrml h5 nii mgz mgh nrrd nhdr mha mhd
            seq trk tck trx tt gii srf mz3 puls resp ecg ext acq pta tfm lta v16 fmr stc real
            cplx short
            mif mnc par rec brik head afni jnii bnii vti vtk gipl fdf fre idf int2 spr sdt
            ft1 ft3 ft4 pipe 1d sec csdf spe h2o lab sin job0 gep tsv
            trf vmr vtc vmp smp voi mtc poi glm ssm prt sdm fib tsf
            label annot curv sulc thickness inflated pial white sphere orig smoothwm
            exar1 examcard pro mode ymf phantom fts)
TYPES=(io.github.julianmer.miox.mr-data org.nema.dicom gov.nih.nifti-1 org.gnu.gnu-zip-archive
       org.napari.napari.h5 com.mathworks.workspace public.data)

QUICK=0 INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --install) INSTALL=1 ;;
        *) echo "unknown option $arg" >&2; exit 1 ;;
    esac
done
if [[ $QUICK == 1 && -x "$APPEX/Contents/Resources/python/bin/python3" ]]; then
    echo "== Keeping the bundled Python"
else
    QUICK=0
    rm -rf "$BUILD"
fi
mkdir -p "$APP/Contents/MacOS" "$APPEX/Contents/MacOS" "$APPEX/Contents/Resources"

if [[ $QUICK == 0 ]]; then
echo "== Python $PYTHON_VERSION with MIOX"
uv python install "$PYTHON_VERSION" --install-dir "$BUILD/pythons" >/dev/null
cp -R "$(ls -d "$BUILD"/pythons/cpython-"$PYTHON_VERSION"*/ | head -1)" "$APPEX/Contents/Resources/python"
PY="$APPEX/Contents/Resources/python/bin/python3"
uv pip install --quiet --python "$PY" --break-system-packages "$ROOT"
rm -rf "$BUILD/pythons"
# what a preview never runs: test suites, installers, Tk and IDLE, C headers
LIB="$APPEX/Contents/Resources/python/lib"
find "$LIB/python$PYTHON_VERSION/site-packages" -type d \( -name tests -o -name test \) -prune \
    -exec rm -rf {} +
rm -rf "$LIB/python$PYTHON_VERSION"/{test,idlelib,tkinter,turtledemo,ensurepip} \
    "$LIB/python$PYTHON_VERSION/site-packages"/pip "$LIB/python$PYTHON_VERSION/site-packages"/pip-* \
    "$LIB"/tcl9* "$LIB"/tk9* "$LIB"/itcl* "$LIB"/thread* "$LIB"/libtcl* \
    "$APPEX/Contents/Resources/python/include"
# the sandbox cannot write bytecode into the bundle: compile it now, valid whatever the file times
"$PY" -m compileall -q -j 0 --invalidation-mode unchecked-hash "$APPEX/Contents/Resources/python/lib" \
    >/dev/null || true
fi

echo "== Swift"
plist() {  # Info.plist with the version, extensions and types filled in
    local items="$2" indent="$3" text=""
    for item in $items; do text+="$indent<string>$item</string>\n"; done
    sed -e "s/__VERSION__/$VERSION/" -e "s/__SDK__/$(xcrun --sdk macosx --show-sdk-version)/g" \
        -e "s/__SDKBUILD__/$(xcrun --sdk macosx --show-sdk-build-version)/" "$1" | awk -v key="$4" -v text="$text" \
        '$0 ~ key { printf "%s", text; next } { print }'
}
plist "$SRC/App/Info.plist" "${EXTENSIONS[*]}" "                    " "__EXTENSIONS__" \
    > "$APP/Contents/Info.plist"
plist "$SRC/QuickLook/Info.plist" "${TYPES[*]}" "                " "__TYPES__" \
    > "$APPEX/Contents/Info.plist"
swiftc -O -target "$TARGET" -o "$APP/Contents/MacOS/MIOX" "$SRC/App/main.swift"
swiftc -O -target "$TARGET" -parse-as-library -application-extension \
    -module-name MIOXQuickLook -framework QuickLookUI \
    -Xlinker -e -Xlinker _NSExtensionMain \
    -o "$APPEX/Contents/MacOS/MIOXQuickLook" "$SRC/QuickLook/PreviewProvider.swift"

echo "== Signing (ad hoc, this Mac only)"
codesign --force --sign - --entitlements "$SRC/QuickLook/QuickLook.entitlements" "$APPEX"
codesign --force --sign - "$APP"
du -sh "$APP"

if [[ $INSTALL == 1 ]]; then
    echo "== Installing to ~/Applications"
    mkdir -p "$HOME/Applications"
    rm -rf "$HOME/Applications/MIOX.app"
    ditto "$APP" "$HOME/Applications/MIOX.app"
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
        -f "$HOME/Applications/MIOX.app"
    pluginkit -a "$HOME/Applications/MIOX.app/Contents/PlugIns/MIOXQuickLook.appex"
    pluginkit -e use -i io.github.julianmer.miox.quicklook
    qlmanage -r >/dev/null && qlmanage -r cache >/dev/null
    echo "Installed. Select a spectroscopy file in Finder and press space."
fi
