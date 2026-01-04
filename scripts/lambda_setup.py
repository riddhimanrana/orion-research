#!/usr/bin/env python3
"""
Lambda AI Setup Script for Orion
================================

Automates Lambda AI instance setup:
1. Creates SSH configuration in ~/.ssh/config
2. Installs Orion from GitHub
3. Sets up Python environment
4. Validates setup
5. Provides quick-connect instructions

Usage:
    python scripts/lambda_setup.py --api-key <YOUR-API-KEY> --region us-west-1
    
Or manually:
    # 1. Launch instance on Lambda console
    # 2. Run: bash lambda_init.sh <INSTANCE-IP>
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


LAMBDA_INIT_SCRIPT = """#!/bin/bash
# Lambda AI Instance Setup Script for Orion Research
# Usage: bash lambda_init.sh

set -e  # Exit on error

INSTANCE_IP="${1:-localhost}"
SSH_KEY="$HOME/.ssh/lambda_orion_key"

echo "=========================================="
echo "Orion Lambda AI Setup"
echo "=========================================="
echo "Instance: $INSTANCE_IP"
echo "SSH Key: $SSH_KEY"
echo ""

# 1. Wait for instance to be ready
echo "Waiting for instance to be ready..."
for i in {1..30}; do
    if ssh -i "$SSH_KEY" -o ConnectTimeout=2 -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP "echo OK" >/dev/null 2>&1; then
        echo "✓ Instance ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ Instance timeout after 60 seconds"
        exit 1
    fi
    sleep 2
done

# 2. Install dependencies
echo ""
echo "Installing system dependencies..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential python3-dev 2>/dev/null
echo "✓ System dependencies installed"
EOF

# 3. Clone Orion repo
echo ""
echo "Cloning Orion repository..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
if [ ! -d ~/orion-research ]; then
    git clone https://github.com/riddhimanrana/orion-research.git ~/orion-research
    echo "✓ Repository cloned"
else
    echo "✓ Repository already exists"
fi
EOF

# 4. Create Python environment
echo ""
echo "Setting up Python environment..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
set -e
cd ~/orion-research

# Use system Python 3.10+ or conda
if command -v python3 &> /dev/null; then
    PYTHON=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
    if [ "$PYTHON" = "3.10" ] || [ "$PYTHON" = "3.11" ] || [ "$PYTHON" = "3.12" ]; then
        echo "✓ Python $PYTHON available (using system)"
        python3 -m venv venv || echo "venv already exists"
        source venv/bin/activate
    else
        echo "⚠ Python $PYTHON detected, installing conda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
        ~/miniconda/bin/conda create -y -n orion python=3.11 2>/dev/null
        source ~/miniconda/bin/activate orion
        echo "✓ Conda environment created"
    fi
else
    echo "✗ Python not found"
    exit 1
fi
EOF

# 5. Install Orion
echo ""
echo "Installing Orion dependencies..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
set -e
cd ~/orion-research
source venv/bin/activate 2>/dev/null || source ~/miniconda/bin/activate orion 2>/dev/null || true
pip install -q --upgrade pip wheel setuptools
pip install -e .[all] -q
echo "✓ Orion installed"
EOF

# 6. Validate setup
echo ""
echo "Validating setup..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP << 'EOF'
source venv/bin/activate 2>/dev/null || source ~/miniconda/bin/activate orion 2>/dev/null || true
cd ~/orion-research
python3 scripts/setup_validate.py --json
echo "✓ Setup validation complete"
EOF

# 7. Create SSH config entry
echo ""
echo "Creating SSH config entry..."
mkdir -p ~/.ssh
cat >> ~/.ssh/config << EOF_CONFIG
Host lambda-orion
    HostName $INSTANCE_IP
    User ubuntu
    IdentityFile $SSH_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
EOF_CONFIG
echo "✓ SSH config updated"

# 8. Print next steps
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  SSH: ssh lambda-orion"
echo ""
echo "  Run pipeline:"
echo "    ssh lambda-orion 'cd ~/orion-research && source venv/bin/activate && python -m orion.cli.run_showcase --episode demo'"
echo ""
echo "  VS Code Remote SSH:"
echo "    1. Install 'Remote - SSH' extension"
echo "    2. Connect to 'lambda-orion' from command palette"
echo "    3. Open folder: /home/ubuntu/orion-research"
echo ""
echo "  Monitor GPU:"
echo "    ssh lambda-orion nvidia-smi"
echo ""
"""

SSH_CONFIG_TEMPLATE = """
# Lambda AI Orion Instance
Host lambda-orion
    HostName {instance_ip}
    User ubuntu
    IdentityFile ~/.ssh/lambda_orion_key
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    
    # Port forwarding for Jupyter
    LocalForward 8888 127.0.0.1:8888
    
    # Port forwarding for Memgraph
    LocalForward 7687 127.0.0.1:7687
"""

VSCODE_SETTINGS = """
{{
    "python.defaultInterpreterPath": "/home/ubuntu/orion-research/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "[python]": {{
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {{
            "source.organizeImports": true
        }}
    }},
    "files.exclude": {{
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.*": true
    }}
}}
"""


class LambdaSetup:
    """Manages Lambda AI Orion setup."""

    def __init__(self, api_key: Optional[str] = None, region: str = "us-west-1"):
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")
        self.region = region
        self.ssh_key_path = Path.home() / ".ssh" / "lambda_orion_key"
        self.ssh_config_path = Path.home() / ".ssh" / "config"

    def create_lambda_init_script(self, output_path: str = "lambda_init.sh"):
        """Create standalone Lambda init script."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(LAMBDA_INIT_SCRIPT)
        os.chmod(output_path, 0o755)
        logger.info(f"✓ Created {output_path}")
        logger.info(f"  Usage: bash {output_path} <INSTANCE-IP>")

    def setup_ssh_config(self, instance_ip: str):
        """Add instance to SSH config."""
        logger.info(f"Setting up SSH config for {instance_ip}...")
        
        self.ssh_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_entry = SSH_CONFIG_TEMPLATE.format(instance_ip=instance_ip)
        
        # Check if already exists
        if self.ssh_config_path.exists():
            with open(self.ssh_config_path) as f:
                content = f.read()
                if "lambda-orion" in content:
                    logger.info("  ✓ SSH config already has lambda-orion entry")
                    return
        
        # Append to config
        with open(self.ssh_config_path, "a") as f:
            f.write(config_entry)
        
        logger.info(f"  ✓ SSH config updated: {self.ssh_config_path}")
        logger.info(f"  Quick connect: ssh lambda-orion")

    def create_vscode_config(self, output_path: str = ".vscode/settings.json"):
        """Create VS Code remote settings."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(VSCODE_SETTINGS)
        logger.info(f"✓ Created {output_path} for remote development")

    def test_connection(self, instance_ip: str) -> bool:
        """Test SSH connection to instance."""
        logger.info(f"Testing SSH connection to {instance_ip}...")
        try:
            result = subprocess.run(
                f"ssh -i {self.ssh_key_path} -o ConnectTimeout=5 ubuntu@{instance_ip} 'echo OK'",
                shell=True, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info("  ✓ Connection successful")
                return True
            else:
                logger.error(f"  ✗ Connection failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"  ✗ Connection error: {e}")
            return False

    def print_startup_guide(self, instance_ip: str):
        """Print startup guide."""
        guide = f"""
========================================
✓ LAMBDA AI SETUP COMPLETE
========================================

Instance IP: {instance_ip}
SSH Key: {self.ssh_key_path}

QUICK START
===========

1. SSH Access:
   ssh lambda-orion

2. First Run:
   ssh lambda-orion << 'EOF'
   cd ~/orion-research
   source venv/bin/activate
   python -m orion.cli.run_showcase --episode test_demo --video data/examples/test.mp4
   EOF

3. VS Code Remote Development:
   a. Install "Remote - SSH" extension in VS Code
   b. Click Remote Explorer icon (bottom left)
   c. Add host: lambda-orion
   d. Open folder: /home/ubuntu/orion-research
   e. Select Python interpreter: ./venv/bin/python
   f. Edit code locally, execute remotely with full GPU power

4. Monitor GPU:
   ssh lambda-orion nvidia-smi -l 1

5. Jupyter Development:
   ssh lambda-orion "cd ~/orion-research && source venv/bin/activate && jupyter lab --ip 0.0.0.0 --no-browser"
   Then open: http://localhost:8888 (token in terminal)

6. File Sync:
   # Download results
   scp -r lambda-orion:~/orion-research/results/ ./results-lambda/
   
   # Upload new videos
   scp video.mp4 lambda-orion:~/orion-research/data/examples/

BEST PRACTICES
==============

• Use git pull to sync code changes
• Keep large datasets on Lambda (use /home/ubuntu/data)
• Store results locally (rsync after runs)
• Monitor GPU: nvidia-smi before/after runs
• Terminate instance when done: Lambda console or API
• Check Lambda dashboard for billing

COST ESTIMATION
===============

GPU: 1x A100 (40GB) = $1.29/hour
• 1 hour development: $1.29
• 8 hours batch processing: $10.32
• Full day continuous: $30.96
• $400 credits ≈ 310 hours (~2 weeks continuous)

NEXT STEPS
==========

1. Validate setup: ssh lambda-orion 'cd ~/orion-research && python scripts/setup_validate.py'
2. Run test: ssh lambda-orion 'cd ~/orion-research && python -m orion.cli.run_showcase --episode test --max-frames 50'
3. Configure VS Code Remote SSH
4. Start development!

========================================
"""
        print(guide)


def main():
    parser = argparse.ArgumentParser(
        description="Setup Lambda AI for Orion development",
        epilog="Examples:\n"
               "  python scripts/lambda_setup.py --create-script\n"
               "  bash lambda_init.sh 1.2.3.4\n"
               "  python scripts/lambda_setup.py --instance-ip 1.2.3.4 --test"
    )
    parser.add_argument("--create-script", action="store_true", help="Create lambda_init.sh script")
    parser.add_argument("--instance-ip", help="Lambda instance IP address")
    parser.add_argument("--region", default="us-west-1", help="Lambda AI region")
    parser.add_argument("--test", action="store_true", help="Test SSH connection")
    parser.add_argument("--vscode", action="store_true", help="Create VS Code config")

    args = parser.parse_args()

    setup = LambdaSetup(region=args.region)

    if args.create_script:
        setup.create_lambda_init_script()
        logger.info("\nNext: bash lambda_init.sh <INSTANCE-IP>")
        return

    if args.instance_ip:
        setup.setup_ssh_config(args.instance_ip)
        
        if args.test:
            setup.test_connection(args.instance_ip)
        
        if args.vscode:
            setup.create_vscode_config()
        
        setup.print_startup_guide(args.instance_ip)
        return

    # Default: create setup script
    setup.create_lambda_init_script()
    print("""
Next steps:

1. Launch instance on Lambda console:
   https://cloud.lambda.ai/instances
   
2. Choose: 1x A100 (40GB) in us-west-1, ~$1.29/hr
   
3. Copy instance IP and run:
   bash lambda_init.sh <INSTANCE-IP>
   
4. Or manually run:
   python scripts/lambda_setup.py --instance-ip <IP> --test --vscode
""")


if __name__ == "__main__":
    main()
