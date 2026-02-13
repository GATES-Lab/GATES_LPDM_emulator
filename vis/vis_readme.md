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
  - [ ] Quiver plot with direction as the arrows and windspeed as colour, annual and seasonal. See example below
  <img width="257" height="342" alt="image" src="https://github.com/user-attachments/assets/23a61b3d-1ec8-44d5-b143-0a0bc914acdc" />

  
  - [ ] Seasonal wind roses - stacked polar bar charts (e.g. see below), showing the frequency of wind speeds and angle as density (not absolute count). Make sure all seasonal plots have the same axis (e.g. in the plot below, the outward circle indicates 20% of the data. Do not take the mean of each datapoint across time. Desired optional inputs would include: 1) user-defined wind-speed bins, 2) plot only specific periods of the day (e.g. plot winds only at midday, or only in the afternoon), 3) atmospheric level to plot. Ideally this a wrapper for a relatively flexible windrose plotting function, so that we can also use it later down the line to plot other things, e.g. the wind distributions at the footprint locations.
  <img width="278" height="123" alt="image" src="https://github.com/user-attachments/assets/dd619d84-bd9b-47cd-a70c-b0daae91f76e" />

  - [ ] Plots of vertical wind, and wind gradients
- [ ] Characterisation of PBLH (research how it is plotted in literature)

## Footprints
- [x]  Distribution of measurements in space and season
- [x]  Mean footprint in space
- [ ]  Mean footprint shape per region and season

## Topography
- [x]  Topography/contour plot of elevation [JC: Not yet brought into main notebook]
- [x]  Land-use map with labels [JC: Not yet brought into main notebook].

| Number label      | Land surface type |
| ----------- | ----------- |
| 0?  | sea? (double check that it is 0! maybe it's nan. either way, we can use this as a sea/land mask)    |
| 1      | broadleaf trees       |
| 2   | needleleaf trees        |
| 3   | C3 (temperate) grass       |
| 4   | C4 (tropical) grass     |
| 5   | shrubs      |
| 6   | urban    |
| 7   | inland water  |
| 8   | bare soil      |
| 9   | ice    |

- [ ]  Histogram of heights and land-uses across the domain [JC: I created a simple bar chart of land-use frequencies]
