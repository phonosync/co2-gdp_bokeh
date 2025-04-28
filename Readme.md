# Sample Bokeh App on the CO2-GDP Dataset
This is a sample Bokeh app that visualizes the CO2-GDP dataset. The app allows users to explore the relationship between CO2 emissions and GDP across different countries and years.

The app is deployed as an Azure App Service and can be accessed at the following URL:
https://co2-gdp-bokeh-db.manuel-doemer.ch

App Service Stack: Python - Version 3.10

Custom domain: co2-gdp-bokeh-db.manuel-doemer.ch

Startup command:
```bash
python -m bokeh serve main.py --port 8000 --address 0.0.0.0 --unused-session-lifetime 5000 --check-unused-sessions 5000 --allow-websocket-origin=<AZURE GENERATED URL>.azurewebsites.net --allow-websocket-origin=co2-gdp-bokeh-db.manuel-doemer.ch
```

GitHub actions workflow - created automatically by Azure App Service:
.github/workflows/main_bokehsampledb.yml


The corresponding `requirements.txt` file is provided for deploying the app on Azure App Service, as it currently does not support conda environments.

`conda.yml` file is provided for setting up a local development environment using Anaconda.

Run the app locally:
```bash
bokeh serve main.py --show
```

Documentation:
- [Bokeh Documentation](https://docs.bokeh.org/en/latest/docs/user_guide/server.html)
- Azure App Service Python WebApp Documentation: [Deploying Python Web Apps](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python)
- CI/CD: [Azure App Service and GitHub Actions](https://learn.microsoft.com/en-us/azure/app-service/deploy-github-actions)
