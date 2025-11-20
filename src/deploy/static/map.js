document.addEventListener("DOMContentLoaded", () => {
  const locations = window.storyLocations || [];

  if (locations.length > 0) {
    const map = L.map("map").setView([locations[0].lat, locations[0].lon], 4);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    const bounds = [];
    locations.forEach((loc) => {
      L.marker([loc.lat, loc.lon])
        .addTo(map)
        .bindPopup(`<b>${loc.name}</b>`)
        .bindTooltip(loc.name, { permanent: false, direction: "top" });
      bounds.push([loc.lat, loc.lon]);
    });

    if (bounds.length > 1) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }
});
