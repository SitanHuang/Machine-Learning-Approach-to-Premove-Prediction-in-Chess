#!/usr/bin/env bash

# Due to chess.com policy, human operators manually selects approximately 100 players across 1000 and above elo players, and scraped games using below command:

for g in $(curl -Ls https://api.chess.com/pub/player/XXX/games/archives | jq -rc ".archives[]") ; do curl -Ls "$g" | jq -rc ".games[].pgn" ; done >> output.pgn