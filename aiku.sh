apt update -y
apt install software-properties-common -y
apt install python3-distutils -y
apt install python3-pip -y
apt install python3-lib2to3 -y
apt install python3-gdbm -y
apt install python3-tk -y
apt install python3-dev -y
apt install python3-venv -y
apt install aptitude -y
apt install xvfb -y
apt install git -y
apt install screen -y
apt install unzip -y
apt install htop -y
apt install nload -y
apt install iftop -y
apt install libxss1 -y
apt install libappindicator1 -y
apt install libindicator7 -y
apt install wget -y
apt install curl -y
apt install iotop -y
apt install ssl-cert -y
apt install openvpn -y
apt install dialog -y
apt install fping -y
apt install nano -y
apt install dnsutils -y
apt install libc6 -y
apt install libstdc++6 -y
apt install libgcc1 -y
apt install libgtk2.0-0 -y
apt install libasound2 -y
apt install libxrender1 -y
apt install libdbus-glib-1-2 -y
apt install xserver-xephyr -y
apt install vnstat -y
apt install net-tools -y
apt install apt-transport-https -y
apt install ca-certificates -y
apt install gnupg-agent -y
apt install lsb-release -y
apt install jq -y
python3 -m pip install markdownify lxml python-wordpress-xmlrpc yake requests newspaper3k lxml_html_clean
python3 -m pip install transformers==4.42.3
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python3 -m pip install -U git+https://github.com/anbuhckr/md2html.git
python3 -m pip install -U "huggingface_hub[cli]"
huggingface-cli login --token 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
mkdir llama
mkdir llama/text-rewriter
huggingface-cli download Ateeqq/Text-Rewriter-Paraphraser --local-dir ./llama/text-rewriter
python3 -m pip install pytest-playwright playwright -U
playwright install-deps
playwright install
apt install -f -y
apt --fix-broken install -y
dd if=/dev/zero of=/swap.file bs=1G count=16
chmod 600 /swap.file
mkswap /swap.file
swapon /swap.file
echo "/swap.file none swap sw 0 0" >> /etc/fstab
apt install -f -y