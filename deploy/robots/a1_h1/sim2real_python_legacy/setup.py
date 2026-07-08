from setuptools import find_packages, setup

package_name = "sim2real_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/a1_policy_bridge.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="woan",
    maintainer_email="woan@example.com",
    description="A1 table-tennis sim2real bridge.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "a1_policy_bridge = sim2real_bridge.a1_policy_bridge_node:main",
        ],
    },
)
