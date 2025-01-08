library(sf)
library(tidyverse)
library(leaflet)

#setwd('Documents/git/SDG/sdg-dashboard/')

nodes <- sf::read_sf('data-raw/MSS_nodes/dsm2_nodes_newcs_extranodes.shp')
channels <- sf::read_sf('data-raw/fc2024.01_chan/FC2024.01_channels_centerlines.shp')
channels_with_numbers <- read_csv('data-raw/channel_names_from_h5.csv')

nodes |> glimpse()
channels |> glimpse()
channels_with_numbers |> glimpse()

unique(channels$id)
unique(channels_with_numbers$...1)
unique(channels_with_numbers$chan_no)

channels_merge <- channels |> 
  left_join(channels_with_numbers |> rename(id = chan_no)) |> 
  sf::st_transform(crs = 4326) |> 
  #filter(!is.na(name)) |> 
  glimpse()

nodes |> group_by(id) |> tally() |> filter(n > 1)

nodes_merge <- nodes |> 
  filter(id != 80) |> 
  left_join(channels_with_numbers |> rename(id = chan_no)) |> 
  filter(!is.na(name)) |> 
  sf::st_transform(crs = 4326) |> 
  glimpse()

leaflet() |> 
  addTiles() |> 
  #addCircleMarkers(data = nodes_merge) |> 
  #addCircleMarkers(data = nodes_merge |> filter(id %in% c(112, 176, 69)))
  addPolylines(data = channels_merge |> filter(id %in% c(211, 79, 134)),
               color = "darkgreen",
               popup = paste0("id = ", channels_merge$id, "<br>",
                              "name = ", channels_merge$name, "<br>",
                              "distance = ", channels_merge$distance, "<br>",
                              "variable = ", channels_merge$variable, "<br>",
                              "interval = ", channels_merge$interval, "<br>",
                              "period_op = ", channels_merge$period_op))

