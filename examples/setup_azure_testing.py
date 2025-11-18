"""
Setup Script for Azure AI Foundry Agent Testing
================================================

This script helps you set up everything needed to test Azure AI Foundry agents.

Usage:
------
python setup_azure_testing.py
"""

import os
import sys
import subprocess


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


def check_environment():
    """Check if required environment variables are set"""
    print_section("📋 Checking Environment Variables")
    
    required_vars = {
        "AZURE_AI_PROJECT_ENDPOINT": "Your Azure AI Foundry project endpoint",
        "AZURE_AI_MODEL_DEPLOYMENT": "Your model deployment name"
    }
    
    optional_vars = {
        "AZURE_AI_API_KEY": "API key (optional if using az login)"
    }
    
    missing = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}")
            print(f"   Value: {value}")
        else:
            print(f"❌ {var} - NOT SET")
            print(f"   Description: {description}")
            missing.append(var)
    
    print()
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var} - SET")
        else:
            print(f"⚪ {var} - Not set (optional)")
            print(f"   Description: {description}")
    
    return missing


def install_packages():
    """Install required packages"""
    print_section("📦 Installing Required Packages")
    
    packages = [
        ("agent-framework-azure-ai", "--pre"),
        ("azure-identity", ""),
        ("azure-ai-projects", "")
    ]
    
    for package, flags in packages:
        print(f"Installing {package}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if flags:
                cmd.append(flags)
            cmd.append(package)
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}")
            print(f"     Error: {e}")
            return False
    
    return True


def test_connection():
    """Test connection to Azure AI Foundry"""
    print_section("🔌 Testing Azure AI Foundry Connection")
    
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT")
    
    if not endpoint or not deployment:
        print("❌ Cannot test connection - environment variables not set")
        return False
    
    print(f"Endpoint: {endpoint}")
    print(f"Model: {deployment}")
    print("\nAttempting connection...")
    
    try:
        # Try to import and create a simple agent
        from agent_framework import ChatAgent
        from agent_framework_azure_ai import AzureAIAgentClient
        from azure.identity.aio import DefaultAzureCredential
        import asyncio
        
        async def quick_test():
            async with ChatAgent(
                chat_client=AzureAIAgentClient(
                    project_endpoint=endpoint,
                    model_deployment_name=deployment,
                    async_credential=DefaultAzureCredential(),
                    agent_name="SetupTestAgent",
                ),
                instructions="You are a test agent.",
            ) as agent:
                result = await agent.run("Say 'test successful' if you can hear me")
                return result.text
        
        response = asyncio.run(quick_test())
        print(f"\n✅ Connection successful!")
        print(f"Agent response: {response}")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure agent-framework-azure-ai is installed with --pre flag")
        return False
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that your endpoint and deployment name are correct")
        print("  2. Ensure you're logged into Azure: az login")
        print("  3. Verify your Azure subscription is active")
        print("  4. Check that the model deployment is running in Azure AI Foundry")
        return False


def provide_setup_instructions(missing_vars):
    """Provide instructions for setting up missing variables"""
    print_section("📝 Setup Instructions")
    
    if missing_vars:
        print("To complete setup, set these environment variables:\n")
        print("PowerShell:")
        for var in missing_vars:
            print(f'  $env:{var} = "your-value-here"')
        
        print("\nLinux/Mac:")
        for var in missing_vars:
            print(f'  export {var}="your-value-here"')
        
        print("\n" + "="*70)
        print("Don't have an Azure AI Foundry project yet?")
        print("="*70)
        print("\nOption 1: Create in Azure Portal")
        print("  1. Go to https://ai.azure.com/")
        print("  2. Create a new project")
        print("  3. Deploy a model (recommended: gpt-4o-mini)")
        print("  4. Copy endpoint and deployment name")
        
        print("\nOption 2: Use GitHub Models (Free)")
        print("  1. Set OPENAI_API_KEY with your GitHub PAT")
        print("  2. Use test_real_agents.py instead")
        print("  3. Test with: python example_test_openai_agent.py")
        
        print("\nFor detailed instructions, see:")
        print("  - azure_ai_foundry_setup.md")
        print("  - AZURE_QUICK_START.md")


def main():
    """Main setup flow"""
    print("="*70)
    print("AZURE AI FOUNDRY AGENT TESTING - SETUP WIZARD")
    print("="*70)
    
    # Step 1: Check environment
    missing = check_environment()
    
    # Step 2: Install packages
    print("\nWould you like to install required packages? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        if not install_packages():
            print("\n❌ Package installation failed")
            return 1
    else:
        print("Skipping package installation")
    
    # Step 3: Test connection (only if env vars are set)
    if not missing:
        print("\nWould you like to test the connection? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            if test_connection():
                print_section("🎉 Setup Complete!")
                print("You're ready to test Azure AI Foundry agents!")
                print("\nNext steps:")
                print("  1. Run: python example_azure_agent.py")
                print("  2. Run: pytest test_azure_ai_agents.py -v")
                print("  3. Create your own custom agents and tools")
                return 0
            else:
                print("\n⚠️  Setup incomplete - connection test failed")
                print("Please check the troubleshooting steps above")
                return 1
    else:
        provide_setup_instructions(missing)
        print("\n⚠️  Setup incomplete - please set environment variables")
        print("After setting them, run this script again to test the connection")
        return 1


if __name__ == "__main__":
    sys.exit(main())
