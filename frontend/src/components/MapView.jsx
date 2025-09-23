import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import { useTranslation } from "react-i18next";
import "leaflet/dist/leaflet.css";

const clusterIconUrls = [
  "http://maps.google.com/mapfiles/ms/icons/red-dot.png",
  "http://maps.google.com/mapfiles/ms/icons/blue-dot.png",
  "http://maps.google.com/mapfiles/ms/icons/green-dot.png",
  "http://maps.google.com/mapfiles/ms/icons/orange-dot.png",
  "http://maps.google.com/mapfiles/ms/icons/purple-dot.png",
];

const getClusterIcon = (clusterId) => {
  if (clusterId === null || clusterId === undefined || clusterId < 0) {
    return L.icon({
      iconUrl: "http://maps.google.com/mapfiles/ms/icons/ltblue-dot.png",
      iconSize: [32, 32],
      iconAnchor: [16, 32],
    });
  }

  const url = clusterIconUrls[clusterId % clusterIconUrls.length];
  return L.icon({
    iconUrl: url,
    iconSize: [32, 32],
    iconAnchor: [16, 32],
  });
};

const MapView = () => {
  const { t } = useTranslation();
  const [complaints, setComplaints] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/complaints")
      .then((res) => res.json())
      .then((data) => setComplaints(data))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="mt-4 flex flex-col items-center">
      <h2 className="text-2xl font-semibold text-green-700 mb-4">{t("map_view")}</h2>
      <div className="w-full max-w-5xl h-[500px] rounded-lg overflow-hidden shadow-lg border border-gray-300">
        <MapContainer center={[30, 78.5]} zoom={5} className="w-full h-full">
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {complaints.map((c) => {
            if (!c.latitude || !c.longitude) return null;

            return (
              <Marker
                key={c.id}
                position={[parseFloat(c.latitude), parseFloat(c.longitude)]}
                icon={getClusterIcon(c.cluster_id)}
              >
                <Popup>
                  <div className="text-gray-800">
                    <p className="font-bold">{t("id")}: <span className="font-normal">{c.id}</span></p>
                    <p className="font-bold">{t("text")}: <span className="font-normal">{c.text}</span></p>
                    <p className="font-bold">{t("cluster")}: <span className="font-normal">{c.cluster_id ?? t("none")}</span></p>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
};

export default MapView;
