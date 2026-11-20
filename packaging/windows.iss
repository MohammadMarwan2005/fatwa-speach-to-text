; Inno Setup recipe for the Windows installer. Installs for the current user only (no admin rights needed).
; Build: iscc /DAppVersion=0.4.0 packaging\windows.iss   (after PyInstaller has filled dist\FatwaTranscript)

#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif

[Setup]
AppId={{6B1E7C2A-3F0D-4C1B-9E55-2A7F0F6E8D41}
AppName=Fatwa Transcript
AppVersion={#AppVersion}
AppPublisher=Fatwa Transcript
DefaultDirName={localappdata}\Programs\Fatwa Transcript
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist-installer
OutputBaseFilename=FatwaTranscript-Setup-windows-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Fatwa Transcript

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\FatwaTranscript\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{userprograms}\Fatwa Transcript"; Filename: "{app}\FatwaTranscript.exe"
Name: "{userdesktop}\Fatwa Transcript"; Filename: "{app}\FatwaTranscript.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FatwaTranscript.exe"; Description: "Open Fatwa Transcript now"; Flags: nowait postinstall skipifsilent
