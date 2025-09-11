#!/usr/bin/env python3
"""
Script to help register the Swarms MCP server with Cline.
This generates the exact configuration needed for Cline MCP integration.
"""

import os
import json
from pathlib import Path

def main():
    current_dir = Path(__file__).parent.absolute()
    
    # Load API key from .env
    env_file = current_dir / '.env'
    api_key = None
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('SWARMS_API_KEY='):
                    api_key = line.split('=', 1)[1]
                    break
    
    if not api_key:
        print("❌ Could not find SWARMS_API_KEY in .env file")
        return 1
    
    # Generate the configuration
    config = {
        "name": "swarms",
        "command": "python",
        "args": [str(current_dir / "start_swarms_mcp.py")],
        "env": {
            "SWARMS_API_KEY": api_key
        }
    }
    
    # Global Cline rules: enable using all available MCP tools when needed.
    global_cline_rules = {
        "useAllAvailableMcpToolsWhenNeeded": True,
        "autoDiscoverMcpTools": True
    }
    
    print("🔧 Swarms MCP Server Configuration for Cline")
    print("=" * 50)
    print()
    print("📋 Option A: Manual Configuration in Cline UI")
    print("1. Open Cline in VS Code")
    print("2. Go to: Cline → MCP Servers → Add MCP Server → 'Local'")
    print("3. Fill in these values:")
    print(f"   - Server Name: {config['name']}")
    print(f"   - Command: {config['command']}")
    print(f"   - Args: {json.dumps(config['args'])}")
    print(f"   - Environment Variables:")
    print(f"     {{")
    print(f"       \"SWARMS_API_KEY\": \"{config['env']['SWARMS_API_KEY']}\"")
    print(f"     }}")
    print()
    
    print("📋 Option B: Command Line Registration")
    print("Copy and run this command in your terminal:")
    print()
    command = f"code --add-mcp '{json.dumps(config)}'"
    print(f"   {command}")
    print()
    
    print("📋 Option C: JSON Configuration (for manual .vscode/settings.json)")
    print("Add this to your .vscode/settings.json under 'cline.mcpServers' and include global rules under 'cline.globalRules':")
    print()
    full_config = {
        "globalClineRules": global_cline_rules,
        "mcpServers": {
            config["name"]: config
        }
    }
    print(json.dumps(full_config, indent=2))
    print()
    
    print("🚀 After Registration:")
    print("1. Restart VS Code or reload the Cline extension")
    print("2. Open a new Cline chat")
    print("3. Test with: 'Use the swarms MCP tool list_agents'")
    print()
    
    print("✅ Your Swarms MCP server is ready for integration!")
    return 0

if __name__ == "__main__":
    exit(main())
