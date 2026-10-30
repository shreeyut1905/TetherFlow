# SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setuptools import find_namespace_packages, setup


def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="tetherflow",
    version="0.1.0",
    description="TetherFlow: kinematic and statistical tethers for any-step flow maps.",
    license="Apache-2.0",
    license_files=("LICENSE", "licenses/**/LICENSE"),
    # The experiment configs use implicit namespace packages.
    packages=find_namespace_packages(
        include=("tetherflow", "tetherflow.*", "scripts", "scripts.*"),
        exclude=("*.__pycache__", "*.__pycache__.*"),
    ),
    package_data={
        "scripts.flowmap_dashboard": ["static/*"],
        "scripts.data": ["prompts.txt"],
    },
    # NumPy 1.x (required below) supports Python only through 3.12.
    python_requires=">=3.10,<3.13",
    install_requires=read_requirements(),
)
