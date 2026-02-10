## Data characterisation notebook
Understanding the distribution of model inputs and outputs

Currently, the notebook loads the base data object: full-domain footprints, met without interpolation, and interpolated domain-wise topography

- [ ]  Add loading of Square cropped object (met and topog are interpolated in time and cropped in space) to allow for cropped metrics (e.g. histogram of topographic height across footprint locations)
- [ ] Add capability to split domain into regions (e.g. by country border, in a grid)
- [ ] Add prior plotting capability

### Domain
- [x] Show domain boundaries on map by data series (met, fp, topo) [JC: Not yet brought into main notebook]
  - [ ] Extend to show multiple domains together on one map
  - [ ] Extend to show boundaries for met and fp overlaid together

### Meteorology
- [ ]  Winds
  - [ ] Quiver plot with direction and windspeed as colour, annual and seasonal
  - [ ] Seasonal wind roses
  - [ ] Plots of vertical wind, and wind gradients
- [ ] Characterisation of PBLH (research how it is plotted in literature)

## Footprints
- [ ]  Distribution of measurements in space and season [Elena already has some functions]
- [ ]  Mean footprint in space
- [ ]  Mean footprint shape per region and season

## Topography
- [x]  Topography/contour plot of elevation [JC: Not yet brought into main notebook]
- [ ]  Land-use map with labels. [Elena to find names of nine categories in JULES/MOSES handbook]
- [ ]  Histogram of heights and land-uses across the domain [JC: I created a simple bar chart of land-use frequencies]