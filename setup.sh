$setup_content = @'
#!/bin/bash
apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
apt-get install -y libsm6
apt-get install -y libxext6
apt-get install -y libxrender-dev
apt-get install -y libgl1
apt-get install -y libharfbuzz0b
apt-get install -y libwebp6
'@

$setup_content | Out-File -FilePath "setup.sh" -Encoding utf8 -Force