# Local AI Infrastructure & Container Stack

Production-grade, containerized AI infrastructure optimized for bare-metal hardware. This repository manages the orchestration, multi-GPU VRAM sharding, and retrieval-augmented generation (RAG) pipeline running on a dedicated homelab node.

---

## Hardware Architecture

* **Host Machine:** Dell Precision T7910 Workstation
* **Compute:** Dual Intel Xeon E5-2697 v4
* **Memory:** 256GB DDR4 ECC RAM
* **Accelerators:** 3x NVIDIA GeForce RTX 3060 12GB GPUs (36GB Total VRAM)
* **Storage:** PCIe-to-NVMe M.2 SSD (Container OS/Fast Cache) + High-Capacity SAS HDDs (Storage Volumes)

---

## Container Stack Architecture

| Service | Role | Key Configuration |
| :--- | :--- | :--- |
| **Ollama** | Local Inference Engine | Multi-GPU layer distribution across 3x RTX 3060s |
| **ChromaDB** | Vector Database | Persistent local volume mount for RAG storage |
| **Odysseus** | Agentic Frontend / UI | Web interface with API routing to local engines |

---

## Multi-GPU VRAM Sharding

To run 32B+ open-weight models without host RAM fallback penalties, Ollama dynamically shards network layers across all three GPU acceleration cards.

Ensure the NVIDIA Container Toolkit is installed on the host, and `docker-compose.yml` grants full GPU device access:

```yaml
services:
  ollama:
    image: ollama/ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
