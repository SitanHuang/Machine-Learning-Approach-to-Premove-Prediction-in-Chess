const fs = require('fs');
const path = require('path');

const ROOT_DIRS = ['training', 'validation'];
const CONCURRENCY = 32; // limit concurrent file reads for performance

async function listPgnFiles(baseDir) {
  const dirPath = path.join(__dirname, baseDir);
  let entries;
  try {
    entries = await fs.promises.readdir(dirPath, { withFileTypes: true });
  } catch (err) {
    if (err.code === 'ENOENT') {
      return [];
    }
    throw err;
  }

  return entries
    .filter((e) => e.isFile() && e.name.endsWith('.pgn'))
    .map((e) => path.join(dirPath, e.name));
}

function extractPlayer(content, tagName) {
  const re = new RegExp(`^\\[${tagName}\\s+"([^"]+)"\\]`, 'm');
  const m = content.match(re);
  return m ? m[1].trim() : null;
}

async function processFile(filePath, playerStats) {
  let content;
  try {
    content = await fs.promises.readFile(filePath, 'utf8');
  } catch (err) {
    console.error(`Failed to read ${filePath}:`, err.message);
    return false;
  }

  const white = extractPlayer(content, 'White');
  const black = extractPlayer(content, 'Black');

  if (!white && !black) {
    // No usable tags, skip this file
    return false;
  }

  const seenThisGame = new Set();

  if (white) seenThisGame.add(white);
  if (black) seenThisGame.add(black);

  for (const name of seenThisGame) {
    if (!playerStats[name]) {
      playerStats[name] = { games: 0 };
    }
    playerStats[name].games += 1;
  }

  return true;
}

async function processFilesWithConcurrency(files, playerStats) {
  let idx = 0;
  let totalGames = 0;

  async function worker() {
    while (true) {
      const myIdx = idx;
      if (myIdx >= files.length) break;
      idx += 1;

      const ok = await processFile(files[myIdx], playerStats);
      if (ok) totalGames += 1;
    }
  }

  const workers = [];
  const workerCount = Math.min(CONCURRENCY, files.length || 1);
  for (let i = 0; i < workerCount; i++) {
    workers.push(worker());
  }

  await Promise.all(workers);
  return totalGames;
}

async function main() {
  // Collect all .pgn files from training and validation
  const allFilesArrays = await Promise.all(ROOT_DIRS.map((d) => listPgnFiles(d)));
  const allFiles = allFilesArrays.flat();

  if (allFiles.length === 0) {
    console.log('No .pgn PGN files found under ./training or ./validation');
    return;
  }

  const playerStats = Object.create(null);
  const totalGames = await processFilesWithConcurrency(allFiles, playerStats);
  const uniquePlayers = Object.keys(playerStats).length;

  console.log(`Total games: ${totalGames}`);
  console.log(`Unique players: ${uniquePlayers}`);
  console.log('');
  console.log('Player,TotalGames,RelativeFrequency');

  const sorted = Object.entries(playerStats).sort((a, b) => {
    // sort by descending game count, then name
    if (b[1].games !== a[1].games) {
      return b[1].games - a[1].games;
    }
    return a[0].localeCompare(b[0]);
  });

  for (const [name, info] of sorted) {
    const rel = totalGames > 0 ? info.games / totalGames : 0;
    console.log(`${name},${info.games},${rel.toFixed(6)}`);
  }
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exitCode = 1;
});
