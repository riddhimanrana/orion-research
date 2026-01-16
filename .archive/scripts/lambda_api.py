#!/usr/bin/env python3
"""
Lambda AI API Client for Orion
===============================

Automate Lambda AI instance lifecycle:
- List instances
- Launch instance
- Monitor instance
- Terminate instance
- Sync code/results

Usage:
    python scripts/lambda_api.py list
    python scripts/lambda_api.py launch --gpu a100 --region us-west-1 --name orion-dev
    python scripts/lambda_api.py terminate --instance-id <ID>
    python scripts/lambda_api.py status --instance-id <ID>
    python scripts/lambda_api.py sync-code --instance-id <ID>
    python scripts/lambda_api.py sync-results --instance-id <ID> --local-dir ./results
"""

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Lambda AI API configuration
LAMBDA_API_URL = "https://cloud.lambda.ai/api/v1"
TIMEOUT = 30


@dataclass
class Instance:
    """Lambda AI instance representation."""
    id: str
    name: str
    status: str
    ip: str
    gpu_type: str
    region: str
    cost_per_hour: float

    def __str__(self):
        return f"{self.name:20} | {self.status:10} | {self.ip:15} | {self.gpu_type:20} | ${self.cost_per_hour:6.2f}/hr"


class LambdaAPIClient:
    """Lambda AI REST API client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LAMBDA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LAMBDA_API_KEY not set. "
                "Set via: export LAMBDA_API_KEY=<YOUR-KEY>"
            )
        self.auth = (self.api_key, "")
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request to Lambda API."""
        url = f"{LAMBDA_API_URL}{endpoint}"
        kwargs.setdefault("timeout", TIMEOUT)
        kwargs.setdefault("auth", self.auth)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise

    def list_instances(self) -> List[Instance]:
        """List all instances."""
        logger.info("Fetching instances...")
        data = self._request("GET", "/instances")
        
        instances = []
        for item in data.get("data", []):
            instance = Instance(
                id=item["id"],
                name=item.get("name", "unnamed"),
                status=item.get("status", "unknown"),
                ip=item.get("ip", "N/A"),
                gpu_type=item.get("instance_type", {}).get("description", "N/A"),
                region=item.get("region", {}).get("name", "N/A"),
                cost_per_hour=float(item.get("instance_type", {}).get("price_cents_per_hour", 0) / 100)
            )
            instances.append(instance)
        
        return instances

    def launch_instance(
        self,
        gpu_type: str = "a100",
        region: str = "us-west-1",
        name: str = "orion-dev",
        ssh_key_name: Optional[str] = None
    ) -> Instance:
        """Launch new instance."""
        # Map friendly names to Lambda AI GPU types
        gpu_map = {
            "a100": "gpu_1x_a100_40gb",
            "a100_80gb": "gpu_1x_a100_80gb",
            "h100": "gpu_1x_h100_sxm",
            "v100": "gpu_1x_v100",
            "b200": "gpu_1x_b200_sxm6",
        }
        
        instance_type = gpu_map.get(gpu_type.lower(), gpu_type)
        
        logger.info(f"Launching {instance_type} in {region}...")
        
        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "name": name,
        }
        
        if ssh_key_name:
            payload["ssh_key_names"] = [ssh_key_name]
        
        data = self._request("POST", "/instance-operations/launch", json=payload)
        
        instance_data = data.get("data", {})
        instance = Instance(
            id=instance_data["id"],
            name=instance_data.get("name", name),
            status=instance_data.get("status", "launching"),
            ip=instance_data.get("ip", "pending"),
            gpu_type=instance_type,
            region=region,
            cost_per_hour=float(instance_data.get("instance_type", {}).get("price_cents_per_hour", 0) / 100)
        )
        
        logger.info(f"✓ Instance launched: {instance}")
        return instance

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate instance."""
        logger.info(f"Terminating instance {instance_id}...")
        
        payload = {"instance_ids": [instance_id]}
        self._request("POST", "/instance-operations/terminate", json=payload)
        
        logger.info(f"✓ Instance {instance_id} terminated")
        return True

    def get_instance(self, instance_id: str) -> Instance:
        """Get instance details."""
        instances = self.list_instances()
        for inst in instances:
            if inst.id == instance_id:
                return inst
        raise ValueError(f"Instance {instance_id} not found")

    def wait_for_instance(self, instance_id: str, timeout: int = 300) -> Instance:
        """Wait for instance to be running and have IP."""
        import time
        
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        
        start = time.time()
        while time.time() - start < timeout:
            instance = self.get_instance(instance_id)
            logger.info(f"  Status: {instance.status}, IP: {instance.ip}")
            
            if instance.status == "active" and instance.ip != "N/A":
                logger.info(f"✓ Instance ready: {instance.ip}")
                return instance
            
            time.sleep(5)
        
        raise TimeoutError(f"Instance {instance_id} not ready after {timeout}s")


class OrionLambdaManager:
    """High-level Orion + Lambda AI management."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = LambdaAPIClient(api_key)

    def quick_launch(self, gpu: str = "a100", region: str = "us-west-1") -> str:
        """Launch and setup instance with one command."""
        # Launch
        instance = self.client.launch_instance(gpu_type=gpu, region=region)
        
        # Wait for IP
        instance = self.client.wait_for_instance(instance.id)
        
        logger.info(f"\n✓ Instance ready!")
        logger.info(f"  SSH: ssh -i ~/.ssh/lambda_key.pem ubuntu@{instance.ip}")
        logger.info(f"  Or setup: python scripts/lambda_setup.py --instance-ip {instance.ip}")
        
        return instance.ip

    def quick_terminate(self, instance_id: str) -> bool:
        """Terminate instance."""
        instance = self.client.get_instance(instance_id)
        logger.info(f"Terminating: {instance}")
        return self.client.terminate_instance(instance_id)

    def sync_code(self, instance_ip: str):
        """Sync local code to remote instance."""
        logger.info(f"Syncing code to {instance_ip}...")
        
        # Git push/pull is preferred, but rsync as fallback
        cmd = f"""
        rsync -av --delete \\
            --exclude=.git \\
            --exclude=.venv \\
            --exclude=venv \\
            --exclude=__pycache__ \\
            --exclude=results/ \\
            --exclude=models/ \\
            . ubuntu@{instance_ip}:~/orion-research/
        """
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ Code synced")
        else:
            logger.error(f"✗ Sync failed: {result.stderr}")

    def sync_results(self, instance_ip: str, local_dir: str = "./results-lambda"):
        """Sync results from remote to local."""
        logger.info(f"Downloading results from {instance_ip}...")
        
        os.makedirs(local_dir, exist_ok=True)
        cmd = f"scp -r ubuntu@{instance_ip}:~/orion-research/results/ {local_dir}/"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ Results saved to {local_dir}")
        else:
            logger.error(f"✗ Download failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Lambda AI instance manager for Orion",
        epilog="Environment: export LAMBDA_API_KEY=<YOUR-KEY>"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List instances
    list_parser = subparsers.add_parser("list", help="List all instances")
    
    # Launch instance
    launch_parser = subparsers.add_parser("launch", help="Launch new instance")
    launch_parser.add_argument("--gpu", default="a100", 
                              choices=["a100", "a100_80gb", "h100", "v100", "b200"],
                              help="GPU type")
    launch_parser.add_argument("--region", default="us-west-1", help="Region")
    launch_parser.add_argument("--name", default="orion-dev", help="Instance name")
    launch_parser.add_argument("--wait", action="store_true", help="Wait for IP")
    
    # Terminate instance
    term_parser = subparsers.add_parser("terminate", help="Terminate instance")
    term_parser.add_argument("--instance-id", required=True, help="Instance ID")
    
    # Get instance status
    status_parser = subparsers.add_parser("status", help="Get instance status")
    status_parser.add_argument("--instance-id", required=True, help="Instance ID")
    
    # Sync code
    sync_code_parser = subparsers.add_parser("sync-code", help="Sync code to instance")
    sync_code_parser.add_argument("--instance-id", required=True, help="Instance ID")
    
    # Sync results
    sync_results_parser = subparsers.add_parser("sync-results", help="Sync results from instance")
    sync_results_parser.add_argument("--instance-id", required=True, help="Instance ID")
    sync_results_parser.add_argument("--local-dir", default="./results-lambda", help="Local directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        manager = OrionLambdaManager()
        client = manager.client
        
        if args.command == "list":
            instances = client.list_instances()
            if instances:
                logger.info("Instances:")
                logger.info(f"{'Name':20} | {'Status':10} | {'IP':15} | {'GPU':20} | {'Cost':8}")
                logger.info("-" * 80)
                for inst in instances:
                    print(inst)
            else:
                logger.info("No instances found")
        
        elif args.command == "launch":
            instance = client.launch_instance(
                gpu_type=args.gpu,
                region=args.region,
                name=args.name
            )
            
            if args.wait:
                instance = client.wait_for_instance(instance.id)
                logger.info(f"\n✓ Instance ready: {instance.ip}")
            
        elif args.command == "terminate":
            client.terminate_instance(args.instance_id)
        
        elif args.command == "status":
            instance = client.get_instance(args.instance_id)
            logger.info(f"Status: {instance}")
        
        elif args.command == "sync-code":
            instance = client.get_instance(args.instance_id)
            manager.sync_code(instance.ip)
        
        elif args.command == "sync-results":
            instance = client.get_instance(args.instance_id)
            manager.sync_results(instance.ip, args.local_dir)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
