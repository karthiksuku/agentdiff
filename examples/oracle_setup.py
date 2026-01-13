"""
Example: Setting up Oracle Autonomous Database storage.

This example demonstrates:
- Connecting to Oracle Autonomous Database
- Initializing the schema
- Using Oracle as the trace storage backend
"""

import asyncio
import os

import sys
sys.path.insert(0, "../src")

from agentdiff import configure
from agentdiff.storage import OracleAutonomousStore, create_oracle_store
from agentdiff.core.span import SpanType


async def main():
    """
    Oracle Autonomous Database Setup Example.

    Before running this example, ensure you have:
    1. An Oracle Autonomous Database instance
    2. Downloaded the wallet files
    3. Set the environment variables or configured the wallet path

    Environment variables needed:
    - ORACLE_USER: Database username
    - ORACLE_PASSWORD: Database password
    - ORACLE_DSN: TNS name from tnsnames.ora
    - ORACLE_WALLET_PASSWORD: (optional) Wallet password if encrypted
    """

    print("Oracle Autonomous Database Setup")
    print("=" * 50)

    # Option 1: Using environment variables
    print("\nOption 1: Using environment variables")
    print("-" * 40)
    print("Set these environment variables:")
    print("  export ORACLE_USER=admin")
    print("  export ORACLE_PASSWORD=YourPassword123!")
    print("  export ORACLE_DSN=mydb_high")
    print()

    # Check if environment variables are set
    if all([
        os.getenv("ORACLE_USER"),
        os.getenv("ORACLE_PASSWORD"),
        os.getenv("ORACLE_DSN"),
    ]):
        print("Environment variables detected. Connecting...")

        try:
            # Create store using environment variables
            storage = create_oracle_store(use_env=True)
            await storage.initialize()

            print("Successfully connected to Oracle Autonomous Database!")

            # Run a simple test
            tracer = configure(storage)

            @tracer.trace(name="oracle-test", version="1.0.0")
            async def test_agent():
                with tracer.span("test-span", span_type=SpanType.CUSTOM):
                    return "Hello from Oracle!"

            result = await test_agent()
            print(f"Test result: {result}")

            # Show stats
            stats = await storage.get_statistics()
            print(f"\nDatabase statistics:")
            print(f"  Traces: {stats.get('trace_count', 0)}")
            print(f"  Spans: {stats.get('span_count', 0)}")
            print(f"  Total tokens: {stats.get('total_tokens', 0)}")
            print(f"  Total cost: ${stats.get('total_cost', 0):.4f}")

            await storage.close()

        except Exception as e:
            print(f"Connection failed: {e}")

    else:
        print("Environment variables not set. Showing configuration examples...")

    # Option 2: Using wallet path
    print("\nOption 2: Using wallet path")
    print("-" * 40)
    print("Download your wallet and extract it:")
    print("  unzip Wallet_mydb.zip -d /path/to/wallet")
    print()
    print("Then connect using:")
    print("""
    storage = OracleAutonomousStore(
        wallet_location="/path/to/wallet",
        wallet_password="YourWalletPassword",
        user="admin",
        password="YourDBPassword",
        dsn="mydb_high",
    )
    """)

    # Option 3: Direct connection (for on-premise Oracle)
    print("\nOption 3: Direct connection (on-premise)")
    print("-" * 40)
    print("""
    storage = OracleAutonomousStore(
        user="myuser",
        password="mypassword",
        dsn="hostname:1521/service_name",
    )
    """)

    # Schema information
    print("\nSchema Details")
    print("-" * 40)
    print("AgentDiff will create the following tables:")
    print("  - agentdiff_traces: Main trace records")
    print("  - agentdiff_spans: Individual span records")
    print("  - agentdiff_checkpoints: Replay checkpoints")
    print("  - agentdiff_comparisons: Diff results")
    print()
    print("JSON columns are used for flexible data storage.")
    print("If using Oracle 23ai, vector search is available for semantic diff.")


if __name__ == "__main__":
    asyncio.run(main())
