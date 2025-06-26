// Handles state/sector/year selection, dropdowns, table rendering, and D3 map rendering

let allStates = [];
let selectedStates = [];

// Get selected year from radio buttons
function getSelectedYear() {
  const checked = document.querySelector('input[name="year"]:checked');
  return checked ? checked.value : null;
}

// Get CSV file path for selected year
function getYearCSVFile() {
  const year = getSelectedYear();
  if (year === '2324') return '/static/data/year_2023_24.csv';
  if (year === '2223') return '/static/data/year_2022_23.csv';
  return null;
}

// Populate custom dropdown with states
function populateStateDropdown(states) {
  const dropdownOptions = document.getElementById('dropdownOptions');
  if (!dropdownOptions) return;
  dropdownOptions.innerHTML = '';
  if (states.length > 0) {
    const allStatesOption = document.createElement('div');
    allStatesOption.className = 'dropdown-option';
    allStatesOption.textContent = 'All States';
    allStatesOption.onclick = function(e) {
      e.stopPropagation();
      selectedStates = ['All States'];
      updateSelectedStates();
      closeDropdown();
    };
    if (!selectedStates.includes('All States')) dropdownOptions.appendChild(allStatesOption);
    states.forEach(state => {
      // Remove "India" from dropdown
      if (!selectedStates.includes(state) && !selectedStates.includes('All States') && state.toLowerCase() !== 'india') {
        const option = document.createElement('div');
        option.className = 'dropdown-option';
        option.textContent = state;
        option.onclick = function(e) {
          e.stopPropagation();
          selectedStates = selectedStates.filter(s => s !== 'All States' && s !== 'India');
          if (!selectedStates.includes(state)) {
            selectedStates.push(state);
            updateSelectedStates();
          }
          closeDropdown();
        };
        dropdownOptions.appendChild(option);
      }
    });
    document.getElementById('dropdownPlaceholder').textContent = 'Choose states...';
  } else {
    document.getElementById('dropdownPlaceholder').textContent = 'No states available for this year';
  }
  // Show/hide India MPCE button
  showHideIndiaButton();
}

// Show or hide the India MPCE button based on selection
function showHideIndiaButton() {
  const btn = document.getElementById('indiaMpceBtn');
  if (!btn) return;
  // Hide if any state is selected from dropdown (not India)
  if (selectedStates.length === 0) {
    btn.style.display = 'inline-block';
  } else {
    btn.style.display = 'none';
  }
}

// Update selected states UI and dropdown
function updateSelectedStates() {
  const selectedStatesDiv = document.getElementById('selectedStates');
  selectedStatesDiv.innerHTML = '';
  selectedStates.forEach(state => {
    const box = document.createElement('div');
    box.className = 'state-box';
    box.textContent = state;
    const cross = document.createElement('span');
    cross.className = 'remove-cross';
    cross.textContent = '×';
    cross.onclick = function(e) {
      e.stopPropagation();
      selectedStates = selectedStates.filter(s => s !== state);
      updateSelectedStates();
    };
    box.appendChild(cross);
    selectedStatesDiv.appendChild(box);
  });
  document.getElementById('dropdownPlaceholder').textContent = '';
  document.getElementById('clearSelectionContainer').style.display = selectedStates.length ? 'block' : 'none';
  populateStateDropdown(allStates);
  showHideIndiaButton();
}

// India MPCE button click handler
function selectIndiaOnly() {
  selectedStates = ['India'];
  updateSelectedStates();
  closeDropdown();
}
window.selectIndiaOnly = selectIndiaOnly;

// Load states from CSV for the selected year
function loadStatesFromCSV() {
  const csvFile = getYearCSVFile();
  if (!csvFile) {
    allStates = [];
    populateStateDropdown([]);
    document.getElementById('dropdownPlaceholder').textContent = 'Select year first';
    return;
  }
  d3.csv(csvFile).then(data => {
    const stateList = [];
    data.forEach(d => {
      const state = d.StateName && d.StateName.trim();
      if (state && !stateList.includes(state)) stateList.push(state);
    });
    allStates = stateList;
    if (stateList.length > 0) {
      populateStateDropdown(allStates);
    } else {
      populateStateDropdown([]);
      document.getElementById('dropdownPlaceholder').textContent = 'No states available for this year';
    }
  }).catch(() => {
    allStates = [];
    populateStateDropdown([]);
    document.getElementById('dropdownPlaceholder').textContent = 'No states available for this year';
  });
}

// Listen for year radio changes to reload states
document.querySelectorAll('input[name="year"]').forEach(radio => {
  radio.addEventListener('change', function() {
    updateSelectedStates();
    loadStatesFromCSV();
    showHideIndiaButton();
  });
});
loadStatesFromCSV();

// Get selected sectors from checkboxes
function getSelectedSectors() {
  const checked = Array.from(document.querySelectorAll('#sectorCapsule input[type="checkbox"]:checked'));
  return checked.map(cb => cb.value.toLowerCase());
}

// Render the selection table for chosen states/sectors
function renderSelectionTable(sectors, states) {
  const containerId = 'selectionTableContainer';
  let container = document.getElementById(containerId);
  if (!sectors.length || !states.length) {
    if (container) container.innerHTML = '';
    return;
  }
  let useIndia = states.includes('India');
  // If All States is chosen, do not show India
  const displayStates = (states.includes('All States'))
    ? allStates.filter(s => s.toLowerCase() !== 'india')
    : (useIndia ? ['India'] : states);
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.style.margin = '32px auto 18px auto';
    container.style.maxWidth = '900px';
    container.style.background = '#f6fafd';
    container.style.borderRadius = '16px';
    container.style.boxShadow = '0 2px 12px rgba(74,144,226,0.07)';
    container.style.padding = '18px 24px';
    container.style.fontSize = '1.12rem';
    container.style.fontWeight = '500';
    container.style.letterSpacing = '0.5px';
    container.style.textAlign = 'center';
    const mapsContainer = document.getElementById('mapsContainer');
    mapsContainer.parentNode.insertBefore(container, mapsContainer);
  }
  if (!displayStates.length) {
    container.innerHTML = '';
    return;
  }
  // Helper to get MPCE data for selected states/sectors
  function getMPCEDataFor(states, sectors, callback) {
    const csvFile = getYearCSVFile();
    if (!csvFile) {
      callback({});
      return;
    }
    d3.csv(csvFile).then(data => {
      let filtered;
      if (useIndia) {
        filtered = data.filter(d =>
          (d.StateName && d.StateName.trim().toLowerCase() === 'india') &&
          sectors.includes(d.SectorName && d.SectorName.trim().toLowerCase())
        );
      } else {
        filtered = data.filter(d =>
          states.includes(d.StateName && d.StateName.trim()) &&
          sectors.includes(d.SectorName && d.SectorName.trim().toLowerCase())
        );
      }
      const lookup = {};
      filtered.forEach(d => {
        const state = useIndia ? 'India' : d.StateName.trim();
        const sector = d.SectorName.trim().toLowerCase();
        if (!lookup[state]) lookup[state] = {};
        lookup[state][sector] = {
          actual: d.ActualMPCE,
          predicted: d.PredictedMPCE
        };
      });
      callback(lookup);
    }).catch(() => {
      callback({});
    });
  }

  // Build HTML table for selection
  let html = '<table style="margin:0 auto; border-collapse:collapse; font-size:1.08rem;">';
  html += '<tr>';
  html += '<th style="padding:8px 18px; border-bottom:1.5px solid #b5c8f7;">State</th>';
  html += '<th style="padding:8px 18px; border-bottom:1.5px solid #b5c8f7;">Sector</th>';
  html += '<th style="padding:8px 18px; border-bottom:1.5px solid #b5c8f7; text-align:center;">Actual MPCE (&#8377;)</th>';
  html += '<th style="padding:8px 18px; border-bottom:1.5px solid #b5c8f7; text-align:center;">Predicted MPCE (&#8377;)</th>';
  html += '<th style="padding:8px 18px; border-bottom:1.5px solid #b5c8f7; text-align:center;">Absolute Error (%)</th>';
  html += '</tr>';

  getMPCEDataFor(displayStates, sectors, function(mpceLookup) {
    let hasRows = false;
    displayStates.forEach(state => {
      sectors.forEach(sector => {
        const sectorLabel = sector.charAt(0).toUpperCase() + sector.slice(1);
        const mpce = (mpceLookup[state] && mpceLookup[state][sector]) ? mpceLookup[state][sector] : null;
        let errorPercent = '-';
        if (
          mpce &&
          mpce.actual !== undefined && mpce.predicted !== undefined &&
          !isNaN(mpce.actual) && !isNaN(mpce.predicted) &&
          Number(mpce.actual) !== 0
        ) {
          errorPercent = ((Math.abs(Number(mpce.predicted) - Number(mpce.actual)) / Number(mpce.actual)) * 100).toFixed(2) + '%';
        }
        html += '<tr>';
        html += `<td style="padding:8px 18px; color:#0078d7;">${state}</td>`;
        html += `<td style="padding:8px 18px; color:#0078d7;">${sectorLabel}</td>`;
        html += `<td style="padding:8px 18px; text-align:center;">${mpce && mpce.actual && !isNaN(mpce.actual) ? '₹ ' + Number(mpce.actual).toFixed(2) : '-'}</td>`;
        html += `<td style="padding:8px 18px; text-align:center;">${mpce && mpce.predicted && !isNaN(mpce.predicted) ? '₹ ' + Number(mpce.predicted).toFixed(2) : '-'}</td>`;
        html += `<td style="padding:8px 18px; text-align:center;">${errorPercent}</td>`;
        html += '</tr>';
        hasRows = true;
      });
    });
    if (!hasRows) {
      html += `<tr><td colspan="5" style="padding:12px; color:#d32f2f;">No data for selected states/sectors.</td></tr>`;
    }
    html += '</table>';
    container.innerHTML = html;
  });
}

// Form submit handler for sector/state/year selection
document.getElementById('sectorForm').onsubmit = function() {
  const selectedYear = getSelectedYear();
  const selectedSectors = getSelectedSectors();
  const errorMsgId = 'selectionErrorMsg';
  let errorMsg = '';
  if (!selectedYear) {
    errorMsg = 'Please select a year.';
  } else if (!selectedSectors.length && !selectedStates.length) {
    errorMsg = 'Please select at least one sector and one state.';
  } else if (!selectedSectors.length) {
    errorMsg = 'Please select at least one sector.';
  } else if (!selectedStates.length) {
    errorMsg = 'Please select at least one state.';
  }
  let errorElem = document.getElementById(errorMsgId);
  if (!errorElem) {
    errorElem = document.createElement('div');
    errorElem.id = errorMsgId;
    errorElem.style.color = '#d32f2f';
    errorElem.style.fontWeight = '600';
    errorElem.style.margin = '12px 0 0 0';
    errorElem.style.fontSize = '1.08rem';
    const form = document.getElementById('sectorForm');
    form.insertBefore(errorElem, form.querySelector('.sector-submit-btn'));
  }
  errorElem.textContent = errorMsg;
  if (errorMsg) return false;
  errorElem.textContent = '';
  renderSelectionTable(selectedSectors, selectedStates);
  document.getElementById('mapsContainer').style.display = 'block';
  document.getElementById('row-rural').style.display = 'none';
  document.getElementById('row-urban').style.display = 'none';
  document.getElementById('container-actual-rural').classList.remove('active');
  document.getElementById('container-predicted-rural').classList.remove('active');
  document.getElementById('container-actual-urban').classList.remove('active');
  document.getElementById('container-predicted-urban').classList.remove('active');
  if (selectedSectors.includes("rural")) {
    document.getElementById('row-rural').style.display = 'flex';
    document.getElementById('container-actual-rural').classList.add('active');
    document.getElementById('container-predicted-rural').classList.add('active');
  }
  if (selectedSectors.includes("urban")) {
    document.getElementById('row-urban').style.display = 'flex';
    document.getElementById('container-actual-urban').classList.add('active');
    document.getElementById('container-predicted-urban').classList.add('active');
  }
  const tableElem = document.getElementById('selectionTableContainer');
  if (tableElem) tableElem.scrollIntoView({ behavior: 'smooth' });
  if (typeof renderMPCEMaps === "function") renderMPCEMaps(true);
  return false;
};

// Toggle dropdown for state selection
function toggleDropdown() {
  const dropdown = document.getElementById('dropdownOptions');
  dropdown.classList.toggle('show');
}
window.toggleDropdown = toggleDropdown;

// Clear state selection handler
window.clearSelection = function() {
  selectedStates = [];
  updateSelectedStates();
  closeDropdown();
};

// Close dropdown
function closeDropdown() {
  document.getElementById('dropdownOptions').classList.remove('show');
}

// Get year label for display
function getSelectedYearLabel() {
  const year = getSelectedYear();
  if (year === '2223') return '2022-23';
  if (year === '2324') return '2023-24';
  return '';
}

// Render D3 MPCE maps for selected states/sectors/year
function renderMPCEMaps(forceReload) {
  const width = 540, height = 620;
  const mapConfigs = [
    { svgId: "#map1", tooltipId: "#tooltip1", legendId: "#legend1", valueKey: "ActualMPCE_Rural", title: "Actual MPCE (Rural)" },
    { svgId: "#map2", tooltipId: "#tooltip2", legendId: "#legend2", valueKey: "ActualMPCE_Urban", title: "Actual MPCE (Urban)" },
    { svgId: "#map3", tooltipId: "#tooltip3", legendId: "#legend3", valueKey: "PredictedMPCE_Rural", title: "Predicted MPCE (Rural)" },
    { svgId: "#map4", tooltipId: "#tooltip4", legendId: "#legend4", valueKey: "PredictedMPCE_Urban", title: "Predicted MPCE (Urban)" }
  ];
  const projection = d3.geoMercator()
    .center([80, 23])
    .scale(750)
    .translate([width / 2, height / 2]);
  const path = d3.geoPath().projection(projection);

  function normalizeStateName(name) {
    return name
      .toLowerCase()
      .replace(/[\s\.\&\(\)\-]/g, '');
  }

  function drawLegend(containerId, colorScale, minVal, maxVal) {
    const legendHeight = 340, legendWidth = 22;
    const legendSvg = d3.select(containerId)
      .html("")
      .append("svg")
      .attr("width", 90)
      .attr("height", legendHeight + 120);
    const defs = legendSvg.append("defs");
    const gradientId = containerId.replace("#", "") + "-gradient";
    const gradient = defs.append("linearGradient")
      .attr("id", gradientId)
      .attr("x1", "0%").attr("y1", "100%")
      .attr("x2", "0%").attr("y2", "0%");
    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", colorScale(minVal));
    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colorScale(maxVal));
    legendSvg.append("rect")
      .attr("x", 34)
      .attr("y", 60)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", `url(#${gradientId})`)
      .style("stroke", "#333")
      .style("stroke-width", "1");
    legendSvg.append("text")
      .attr("x", 34 + legendWidth / 2)
      .attr("y", 60 + legendHeight + 28)
      .attr("text-anchor", "middle")
      .attr("font-size", "18px")
      .attr("fill", "#133366")
      .attr("font-weight", "bold")
      .text(minVal);
    legendSvg.append("text")
      .attr("x", 34 + legendWidth / 2)
      .attr("y", 52)
      .attr("text-anchor", "middle")
      .attr("font-size", "18px")
      .attr("fill", "#133366")
      .attr("font-weight", "bold")
      .text(maxVal);
    legendSvg.append("text")
      .attr("x", 34 + legendWidth / 2)
      .attr("y", 28)
      .attr("text-anchor", "middle")
      .attr("font-size", "17px")
      .attr("font-weight", "bold")
      .attr("fill", "#133366")
      .text("MPCE");
  }

  function loadAndRenderMaps() {
    const yearLabel = getSelectedYearLabel();
    if (yearLabel) {
      document.querySelector('#container-actual-rural .map-title').textContent = `Actual MPCE (Rural) - ${yearLabel}`;
      document.querySelector('#container-predicted-rural .map-title').textContent = `Predicted MPCE (Rural) - ${yearLabel}`;
      document.querySelector('#container-actual-urban .map-title').textContent = `Actual MPCE (Urban) - ${yearLabel}`;
      document.querySelector('#container-predicted-urban .map-title').textContent = `Predicted MPCE (Urban) - ${yearLabel}`;
    } else {
      document.querySelector('#container-actual-rural .map-title').textContent = `Actual MPCE (Rural)`;
      document.querySelector('#container-predicted-rural .map-title').textContent = `Predicted MPCE (Rural)`;
      document.querySelector('#container-actual-urban .map-title').textContent = `Actual MPCE (Urban)`;
      document.querySelector('#container-predicted-urban .map-title').textContent = `Predicted MPCE (Urban)`;
    }
    mapConfigs.forEach(cfg => {
      d3.select(cfg.svgId).selectAll("*").remove();
      d3.select(cfg.legendId).selectAll("*").remove();
    });
    const useIndia = selectedStates.includes('India');
    const geojsonFile = useIndia ? '/static/data/india-composite.geojson' : '/static/data/india.json';
    const csvFile = getYearCSVFile();
    if (!csvFile) return; // Prevent d3.csv(null)
    Promise.all([
      d3.json(geojsonFile),
      d3.csv(csvFile)
    ]).then(function([geojson, csvData]) {
      let dataByState = {};
      csvData.forEach(d => {
        const normState = normalizeStateName(d.StateName.trim());
        if (!dataByState[normState]) {
          dataByState[normState] = { csvStateName: d.StateName.trim() };
        }
        if (d.SectorName.trim().toLowerCase() === "rural") {
          dataByState[normState].ActualMPCE_Rural = +d.ActualMPCE;
          dataByState[normState].PredictedMPCE_Rural = +d.PredictedMPCE;
        } else if (d.SectorName.trim().toLowerCase() === "urban") {
          dataByState[normState].ActualMPCE_Urban = +d.ActualMPCE;
          dataByState[normState].PredictedMPCE_Urban = +d.PredictedMPCE;
        }
      });
      const allValues = [];
      Object.values(dataByState).forEach(d => {
        ["ActualMPCE_Rural", "ActualMPCE_Urban", "PredictedMPCE_Rural", "PredictedMPCE_Urban"].forEach(key => {
          if (d[key] !== undefined && !isNaN(d[key])) allValues.push(d[key]);
        });
      });
      const minVal = Math.floor(d3.min(allValues)/ 100) * 100;
      const maxVal = Math.ceil(d3.max(allValues)  / 100) * 100;
      const colorScale = d3.scaleLinear()
        .domain([minVal, maxVal])
        .range(["#fefded", "#ff0000"]);
      const selected = useIndia
        ? ['India']
        : (selectedStates.includes('All States') || selectedStates.length === 0)
          ? Object.values(dataByState).map(d => d.csvStateName)
          : selectedStates;
      function isStateSelected(stateName) {
        if (useIndia) return stateName.trim().toLowerCase() === 'india';
        const norm = normalizeStateName(stateName);
        return selected.some(sel => normalizeStateName(sel) === norm);
      }
      mapConfigs.forEach(cfg => {
        drawLegend(cfg.legendId, colorScale, minVal, maxVal);
        const svg = d3.select(cfg.svgId)
          .attr("width", width)
          .attr("height", height);
        const tooltip = d3.select(cfg.tooltipId);
        svg.selectAll("path")
          .data(geojson.features)
          .enter().append("path")
          .attr("class", function(d) {
            const geoState = (d.properties.st_nm || d.properties.name || "Unknown");
            return isStateSelected(geoState) ? "state state-selected" : "state state-unselected";
          })
          .attr("d", path)
          .attr("fill", function(d) {
            const geoState = (d.properties.st_nm ||  d.properties.name || "Unknown");
            const normGeoState = normalizeStateName(geoState);
            const stateData = dataByState[normGeoState];
            let value = stateData ? stateData[cfg.valueKey] : undefined;
            if (isStateSelected(geoState)) {
              return (value !== undefined && !isNaN(value)) ? colorScale(value) : "#eee";
            } else {
              return "#f2f2f2";
            }
          })
          .attr("opacity", function(d) {
            const geoState = (d.properties.st_nm || d.properties.ST_NM || d.properties.name || "Unknown");
            return isStateSelected(geoState) ? 1 : 0.35;
          })
          .on("mouseover", function(event, d) {
            const geoState = (d.properties.st_nm || d.properties.ST_NM || d.properties.name || "Unknown");
            const normGeoState = normalizeStateName(geoState);
            const stateData = dataByState[normGeoState];
            let value = stateData ? stateData[cfg.valueKey] : "N/A";
            let displayName = stateData ? stateData.csvStateName : geoState;
            let valueDisplay = (value !== undefined && !isNaN(value)) ? `₹ ${Number(value).toFixed(2)}` : value;
            tooltip
              .style("display", "block")
              .html(
                `<strong>${displayName}</strong><br/>` +
                `${cfg.title}: <b>${valueDisplay}</b>`
              );
          })
          .on("mousemove", function(event) {
            tooltip
              .style("left", (event.clientX + 20) + "px")
              .style("top", (event.clientY - 10) + "px");
          })
          .on("mouseout", function() {
            tooltip.style("display", "none");
          });
      });
    });
  }
  if (forceReload) {
    loadAndRenderMaps();
    return;
  }
  loadAndRenderMaps();
}

renderMPCEMaps();

// Attach clear selection handler to button
function attachClearSelectionHandler() {
  const clearBtn = document.getElementById('clear-selection-btn');
  if (clearBtn && !clearBtn._handlerAttached) {
    clearBtn.onclick = function(e) {
      e.stopPropagation();
      selectedStates = [];
      updateSelectedStates();
      closeDropdown();
    };
    clearBtn._handlerAttached = true;
  }
}
attachClearSelectionHandler();
document.addEventListener('DOMContentLoaded', attachClearSelectionHandler);
const observer = new MutationObserver(attachClearSelectionHandler);
observer.observe(document.body, { childList: true, subtree: true });

// Dummy D3 transition for states (if needed)
d3.selectAll(".state")
  .transition()
  .duration(1000)
  .attr("fill", d => colorScale(d.value));