import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class DockerSetupTest(unittest.TestCase):
    def test_downloads_weights_before_build_and_reuses_them(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = root / "repository with spaces"
            scripts = repository / "scripts"
            scripts.mkdir(parents=True)
            shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts" / "setup_docker.sh", scripts / "setup_docker.sh")

            bin_dir = root / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import os, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "with open(os.environ['DOCKER_CALLS'], 'a') as log:\n"
                "    log.write(args[0] + '\\n')\n"
                "if args[0] == 'run':\n"
                "    mount = args[args.index('--mount') + 1]\n"
                "    source = next(part[7:] for part in mount.split(',') if part.startswith('source='))\n"
                "    pathlib.Path(source, 'nomad.pth').write_bytes(b'weights')\n"
            )
            docker.chmod(0o755)
            calls = root / "docker_calls"
            env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}", "DOCKER_CALLS": str(calls)}
            command = ["bash", str(scripts / "setup_docker.sh")]

            subprocess.run(command, check=True, cwd=root, env=env)
            self.assertEqual(calls.read_text().splitlines(), ["run", "build"])
            self.assertEqual((repository / "model_weights" / "nomad.pth").read_bytes(), b"weights")

            subprocess.run(command, check=True, cwd=root, env=env)
            self.assertEqual(calls.read_text().splitlines(), ["run", "build", "build"])


    def test_run_mounts_repository_when_started_elsewhere(self):
        import json

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repository = root / "repository with spaces"
            scripts = repository / "scripts"
            scripts.mkdir(parents=True)
            shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts" / "run_docker.sh", scripts / "run_docker.sh")

            bin_dir = root / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['DOCKER_CALLS'], 'w') as log:\n"
                "    json.dump(sys.argv[1:], log)\n"
            )
            docker.chmod(0o755)
            calls = root / "docker_calls"
            env = os.environ | {
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "DOCKER_CALLS": str(calls),
                "TORCH_VARIANT": "cpu",
            }
            subprocess.run(["bash", str(scripts / "run_docker.sh"), "--group-add", "video"], check=True, cwd=root, env=env)
            args = json.loads(calls.read_text())
            self.assertEqual(args[0], "run")
            self.assertIn(f"type=bind,source={repository},target=/visualnav-transformer", args)
            self.assertNotIn("--gpus=all", args)
            self.assertEqual(args[-3:], ["--group-add", "video", "visualnav_transformer:latest"])

if __name__ == "__main__":
    unittest.main()
