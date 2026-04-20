[Setup]
AppName=Adouga
AppVersion=0.1.0
AppPublisher=Adouga
DefaultDirName={autopf}\Adouga
DefaultGroupName=Adouga
OutputBaseFilename=AdougaSetup
OutputDir=output
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\Adouga.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Adouga"; Filename: "{app}\Adouga.exe"
Name: "{group}\Uninstall Adouga"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Adouga"; Filename: "{app}\Adouga.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\Adouga.exe"; Description: "Launch Adouga"; Flags: nowait postinstall skipifsilent
