"use client";
import { useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

interface AgeModalProps {
  isModalOpen: boolean;
  setIsModalOpen: (open: boolean) => void;
}

const AgeModal: React.FC<AgeModalProps> = ({ isModalOpen, setIsModalOpen }) => {
  const [age, setAge] = useState("");
  const { user } = useUser();
  const [selectedDisease, setSelectedDisease] = useState("");
  const router = useRouter();
  const searchParams = useSearchParams();

  const diseases = [
    { label: "Diabetes", value: "Diabetes" },
    { label: "Hypoglycemia", value: "Hypoglycemia" },
    { label: "Gastroparesis", value: "Gastroparesis" },
    { label: "IBS", value: "IBS" },
    { label: "Peptic Ulcer", value: "Peptic Ulcer" },
    { label: "Hyperthyroidism", value: "Hyperthyroidism" },
    { label: "Kidney Disease", value: "Kidney Disease" },
    { label: "Cystic Fibrosis", value: "Cystic Fibrosis" },
    { label: "Addison's Disease", value: "Addison's Disease" },
  ];

  useEffect(() => {
    // Sync state with URL params if modal re-opens
    const currentDisease = searchParams.get("health_condition");
    if (currentDisease) setSelectedDisease(currentDisease);
  }, [searchParams, isModalOpen]);

  const handleSubmit = async () => {
    // 1. Update User Age if provided
    if (age) {
        try {
            await user?.update({
                unsafeMetadata: {
                    age,
                },
            });
        } catch (e) {
            console.error("Failed to update age", e);
        }
    }

    // 2. Update URL with details
    const params = new URLSearchParams(searchParams.toString());
    
    if (selectedDisease) {
        params.set("health_condition", selectedDisease);
    } else {
        params.delete("health_condition");
    }

    // Reflect simplified "extras" logic
    router.push(`?${params.toString()}`);
    setIsModalOpen(false);
  };

  return (
    <>
      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-5 rounded-2xl shadow-lg w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">
              Enter Your Details
              <button
                onClick={() => setIsModalOpen(false)}
                className="float-right"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </h2>
            {!user?.unsafeMetadata?.age && (
              <>
                <input
                  type="number"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="border rounded-xl p-2 w-full mb-4"
                  placeholder="Your age"
                />
              </>
            )}
            <select
              value={selectedDisease}
              onChange={(e) => setSelectedDisease(e.target.value)}
              className="border rounded-xl p-2 w-full mb-4"
            >
              <option value="">Select a condition (if applicable)</option>
              {diseases.map((disease) => (
                <option key={disease.value} value={disease.value}>
                  {disease.label}
                </option>
              ))}
            </select>
            <button
              onClick={handleSubmit}
              className="bg-red-500 text-white p-2 rounded-xl hover:bg-red-600 transition duration-300 w-full"
            >
              Apply Filter
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default AgeModal;
