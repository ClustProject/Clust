from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='Clust',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        install_requires
    ]
)


    # entry_points={
    #     'console_scripts': [
    #         'clust-cli = clust.cli:main',
    #     ],
    # }


#     # 패키지의 메타데이터 설정
#     author='Your Name',
#     author_email='your.email@example.com',
#     description='A description of your package',
#     url='https://github.com/ClustProject/Clust',
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'License :: OSI Approved :: MIT License',
#         'Operating System :: OS Independent',
#     ],