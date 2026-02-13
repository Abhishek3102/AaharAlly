'use client';

import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import Link from 'next/link';

// Fix Leaflet Default Icon Issue in Next.js
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom Icon for Hotels
const hotelIcon = new L.Icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/1996/1996068.png',
    iconSize: [35, 35],
    iconAnchor: [17, 35],
    popupAnchor: [1, -34],
});


const MapComponent = ({ hotels }: { hotels: any[] }) => {
    // Center map on Mumbai default
    const center: [number, number] = [19.0760, 72.8777]; 

    return (
        <MapContainer center={center} zoom={11} scrollWheelZoom={false} className="h-[500px] w-full rounded-lg shadow-lg z-0">
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {hotels.map((hotel) => (
                <Marker 
                    key={hotel._id} 
                    position={[hotel.location.lat, hotel.location.lng]}
                    icon={hotelIcon}
                >
                    <Popup>
                        <div className="text-center">
                            <img src={hotel.image} alt={hotel.name} className="w-full h-24 object-cover rounded mb-2" />
                            <h3 className="font-bold text-lg">{hotel.name}</h3>
                            <p className="text-sm text-gray-600 mb-2 truncate max-w-[200px]">{hotel.address}</p>
                            <Link href={`/hotels/${hotel._id}`} className="inline-block bg-redCustom text-white text-xs px-3 py-1 rounded hover:opacity-80">
                                View Menu
                            </Link>
                        </div>
                    </Popup>
                </Marker>
            ))}
        </MapContainer>
    );
};

export default MapComponent;
