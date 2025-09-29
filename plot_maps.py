import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr

from constants import PROJECT_DIR


def plot_gaussian_grid(
    data,
    ax=None,
    central_longitude=0,
    title=None,
    cmap="viridis",
    scatter_size=3,
    filename=None,
):
    """Quick plotting function for gaussian grids using scatter plot.

    Parameters:
    -----------
    ds : xarray.Dataset or DataArray
        Dataset containing the variable to plot
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new subplot
    projection : cartopy projection, optional
        Map projection (default: Robinson)
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    scatter_size : int, optional
        Scatter point size
    filename : str, optional
        If provided, save figure to this filename

    Returns:
    --------
    ax : matplotlib axis
    sc : scatter plot object
    """
    # Check required coordinates
    if "latitude" not in data.coords or "longitude" not in data.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates")

    assert "point" in data.dims, "DataArray must have a 'point' dimension"

    assert isinstance(data, xr.DataArray), "Input must be an xarray DataArray"

    if len(data.squeeze().dims) != 1:
        raise ValueError("Only 1 dim (point) must be left after .squeeze()")

    # Get coordinates
    lons = data.longitude.values
    lats = data.latitude.values

    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(
            111,
            projection=ccrs.PlateCarree(central_longitude=central_longitude),
        )

    # Set up map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines()

    # Plot data
    sc = ax.scatter(
        lons,
        lats,
        c=data.values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        s=scatter_size,
    )

    # Add colorbar
    plt.colorbar(sc, ax=ax)

    # Set title
    if title:
        ax.set_title(title)

    # Save figure if filename provided
    if filename:
        figure = ax.get_figure()
        figure.savefig(PROJECT_DIR / filename)
    if ax is None:
        plt.show()
        return fig
    else:
        return None
