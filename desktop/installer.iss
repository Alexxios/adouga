[Setup]
AppName=Adouga Dev
AppVersion=0.1.0
AppPublisher=Adouga
DefaultDirName={autopf}\AdougaDev
DefaultGroupName=Adouga Dev
OutputBaseFilename=AdougaDevSetup
OutputDir=output
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\AdougaDev.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Adouga Dev"; Filename: "{app}\AdougaDev.exe"
Name: "{group}\Uninstall Adouga Dev"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Adouga Dev"; Filename: "{app}\AdougaDev.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\AdougaDev.exe"; Description: "Launch Adouga Dev"; Flags: nowait postinstall skipifsilent
