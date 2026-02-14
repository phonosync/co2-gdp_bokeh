# Sample Bokeh App on the CO2-GDP Dataset

A Bokeh app that uses the CO2-GDP dataset to visualize the relationship between CO2 emissions and GDP.

The app is deployed as an Azure App Service at:
https://co2-gdp-bokeh-db.manuel-doemer.ch

## Data

The dataset is sourced from [Our World in Data – CO2 and Greenhouse Gas Emissions](https://github.com/owid/co2-data/tree/master).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) via `pyproject.toml` and `uv.lock`.

Azure App Service Oryx automatically detects `pyproject.toml` + `uv.lock` and installs dependencies with uv during deployment.

Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up the local environment:
```bash
uv sync
```

Run the app locally:
```bash
uv run bokeh serve main.py --show
```

## Deployment as Azure WebApp

App Service Stack: Python - Version 3.12

GitHub actions workflow:
.github/workflows/main_bokehsampledb.yml

Startup command:
```bash
python -m bokeh serve main.py --port 8000 --address 0.0.0.0 --unused-session-lifetime 5000 --check-unused-sessions 5000 --allow-websocket-origin=<AZURE GENERATED URL>.azurewebsites.net --allow-websocket-origin=co2-gdp-bokeh-db.manuel-doemer.ch
```

Custom domain: co2-gdp-bokeh-db.manuel-doemer.ch

## Documentation
- [Bokeh Documentation](https://docs.bokeh.org/en/latest/docs/user_guide/server.html)
- [uv Documentation](https://docs.astral.sh/uv/)
- Azure App Service Python WebApp Documentation: [Deploying Python Web Apps](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python)
- CI/CD: [Azure App Service and GitHub Actions](https://learn.microsoft.com/en-us/azure/app-service/deploy-github-actions)

## License

This project is licensed under the [CC BY-NC 4.0](LICENSE) license.
