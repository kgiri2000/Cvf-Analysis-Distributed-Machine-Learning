import os
import json
import socket
import pytest
from src.trainer_distributed import setup_tf_config

def test_create_tf_config(monkeypatch):
    #Test that TF_CONFIG is created correctly when none exists
    monkeypatch.delenv("TF_CONFIG", raising= False)
    monkeypatch.setattr(socket, "gethostname", lambda:"localhost")
    monkeypatch.setattr(socket, "gethostbyname", lambda x: "127.0.0.1")

    setup_tf_config()

    assert "TF_CONFIG" in os.environ

    tf_config = json.loads(os.environ["TF_CONFIG"])
    assert "cluster" in tf_config
    assert "task" in tf_config
    assert tf_config["task"]["index"] == 0
    assert tf_config["cluster"]["worker"][0] == "127.0.0.1:12345"


def test_existing_tf_config(monkeypatch):
    #Test the existing TF_CONFIG in env
    existing= {"cluster": {"worker": ["1.1.1.1:12345"]}, "task": {"type": "worker", "index": 2}}
    monkeypatch.setenv("TF_CONFIG", json.dumps(existing))

    result = setup_tf_config()
    assert result == existing

    assert json.loads(os.environ["TF_CONFIG"]) == existing
 
def test_custom_cluster_and_rank(monkeypatch):
    #Test custom  cluster_hosts and rank parameters
    monkeypatch.delenv("TF_CONFIG", raising= False)

    cluster_hosts = ["10.1.1.1:12345", "11.1.1.1.12345"]
    setup_tf_config(cluster_hosts= cluster_hosts, rank = 1)

    tf_config = json.loads(os.environ["TF_CONFIG"])
    assert tf_config["cluster"]["worker"] == cluster_hosts
    assert tf_config["task"]["index"]==1
    
