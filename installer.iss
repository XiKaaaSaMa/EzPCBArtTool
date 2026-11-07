; Inno Setup script for the PCB art board tool.
;
; Packages the PyInstaller directory build into a single installer. The
; directory build is used rather than a one-file executable on purpose: a
; one-file build unpacks several hundred megabytes to a temporary folder on
; every launch, which costs seconds of start-up time for no benefit once an
; installer is doing the unpacking anyway.
;
; Installs per-user by default, so no administrator prompt appears and the
; application never needs write access to its own directory -- it keeps its
; cache and output under %LOCALAPPDATA%, which the code resolves through
; pcbart.paths rather than guessing from __file__.
;
; AppVersion is passed on the command line (/DAppVersion=x.y.z) so a release
; never requires editing this file. See docs/PACKAGING.md for the numbering
; rule: third digit for test builds, second for public beta, first for release.

; Display name. This is what the user reads: Add/Remove Programs entry, Start
; menu folder, shortcut labels, and the "Launch" checkbox. Kept Chinese on
; purpose (user requirement 2026-10-10: 界面仍显示中文名).
#define AppName "PCB艺术板工具"
; File name. Matches the repository name so the installer, the installed .exe
; and the install folder all read the same as the source repo.
;   ⚠️ Do NOT fold this into AppName: they are deliberately different, and
;      AppName drives *labels* while AppDirName / AppExeName drive *paths*.
#define AppDirName "EzPCBArtTool"
; Overridable from the command line:
;   ISCC.exe /DAppVersion=0.1.0 安装包.iss
#ifndef AppVersion
  #define AppVersion "1.0.6"
#endif
#define AppPublisher "PCB Art Board Tool"
#define AppExeName "EzPCBArtTool.exe"
#ifndef AppBuildDir
  #define AppBuildDir "dist\EzPCBArtTool"
#endif
; Both the application and its uninstaller carry this icon.
#define AppIcon "icon\app.ico"

[Setup]
AppId={{7C4A1E3B-9D25-4F81-A6B0-3E5C7D2F8A41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
; Installation folder uses the repo name (file layer); the Start menu folder
; and Add/Remove entry use the Chinese display name above.
DefaultDirName={autopf}\{#AppDirName}
DefaultGroupName={#AppName}
; Per-user install: no elevation prompt, and nothing to go wrong when the
; machine has no administrator available.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=.
OutputBaseFilename=EzPCBArtTool_v{#AppVersion}_安装包
SetupIconFile={#AppIcon}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
; The build tree is ~174 MB; give the progress bar something honest to show.
SetupLogging=yes

[Languages]
; Ships a Simplified Chinese translation alongside the built-in English one,
; so the wizard matches the application. Chinese is first, which makes it
; the default language the installer opens in.
Name: "chinese"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
; The whole PyInstaller output, including the _internal directory that holds
; the Qt, OpenCV and NumPy libraries. The folder name comes from COLLECT(name=)
; in the .spec -- keep AppDirName in sync with it.
Source: "{#AppBuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; GPL-3.0 requires the licence text to travel with the program. Shipped as a
; .txt next to the exe so it is readable without opening the source tree.
Source: "LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

[InstallDelete]
; Dependency layouts can change between PyInstaller/Qt versions. Replacing
; files alone leaves old DLLs beside new extension modules, which can load an
; incompatible runtime on upgrade. This is exclusively program-owned data;
; drafts, image caches and settings live outside the installation directory.
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{group}\许可证（GPL-3.0）"; Filename: "{app}\LICENSE.txt"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: quicklaunchicon

[Run]
; Launch after install, but not silently: the user should choose.
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The per-user cache and generated output. The installer did not create these
; -- the program did at runtime -- so Inno would otherwise leave them behind.
Type: filesandordirs; Name: "{localappdata}\PCB艺术板工具\cache"
Type: filesandordirs; Name: "{localappdata}\PCB艺术板工具\out"
; The draft library is the user's work, so `dirifempty` rather than a delete:
; a library with saved drafts survives the uninstall untouched, while an empty
; one does not leave a stray folder behind. The parent directory gets the same
; treatment, and only disappears once all three children are gone.
Type: dirifempty; Name: "{localappdata}\PCB艺术板工具\drafts"
Type: dirifempty; Name: "{localappdata}\PCB艺术板工具"

; No [Code] section. An earlier version checked for the PyInstaller output in
; InitializeSetup, which was a bug: at run time {src} is the folder the
; installer sits in rather than the build folder, so the check rejected every
; installation. Inno resolves [Files] sources at compile time, so a missing
; build tree already fails the compile with a clear message.
