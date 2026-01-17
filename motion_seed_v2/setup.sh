#!/bin/bash
# Motion Calendar Seed - Setup Script
# For Raspberry Pi deployment

set -e

echo "Motion Calendar Seed - Setup"
echo "============================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install --user -r requirements.txt

# Make CLI executable
chmod +x cli.py

# Create data directory
mkdir -p seed_data

echo ""
echo "Setup complete!"
echo ""
echo "To run:"
echo "  python3 cli.py"
echo ""
echo "Or with custom data path:"
echo "  python3 cli.py /path/to/data"
echo ""
echo "To run on boot (optional):"
echo "  Add to /etc/rc.local:"
echo "  python3 /path/to/motion_seed/cli.py &"
