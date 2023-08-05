from setuptools import setup, find_packages

setup(
    name="sbayes_interactive_maps",
    version="1.0",
    description="plotly-express/dash app for interactive visualization of sBayes cluster results.",
    author="Nico Neureiter",
    author_email="nico.neureiter@gmail.com",
#    long_description=open('README.md').read(),
#    long_description_content_type='text/markdown',
    keywords='data linguistics',
    license='GPLv3',
    url="https://github.com/NicoNeureiter/sbayes_interactive_maps",
    package_dir={'sbayes_dash': 'sbayes_dash'},
    packages=find_packages(),
    platforms='any',
    include_package_data=True,
    package_data={},
    install_requires=[
        "dash",
        "geopandas",
        "matplotlib",
        "jupyter_dash",
        "pandas",
        "numpy",
        "scipy",
        "plotly",
        "pyproj",
        "typing_extensions",
        "Unidecode",
        "gunicorn",
    ],
    entry_points={
        'console_scripts': [
            'sbayes_dash = sbayes_dash.run:main',
        ]
    }
)

