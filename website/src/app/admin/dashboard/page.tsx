"use client";
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useRouter } from 'next/navigation';
import MLControls from '@/components/admin/MLControls';

interface UserStat {
  _id: string;
  email: string;
  joinedAt: string;
  orderCount: number;
  age?: number;
  gender?: string;
  recommendations?: string[];
  scores?: any[];
}

const AdminDashboard = () => {
  const [users, setUsers] = useState<UserStat[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const res = await axios.get('/api/admin/users');
      if (res.data.success) {
        setUsers(res.data.users);
      }
    } catch (error) {
      console.error(error);
      // If unauthorized, redirect to login
      router.push('/admin/login');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    // In a real app, call an API to clear the cookie
    document.cookie = 'admin_token=; Max-Age=0; path=/;';
    router.push('/admin/login');
  };

  if (loading) return <div className="p-10 text-center">Loading Admin Panel...</div>;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Admin Dashboard</h1>
          <div className="flex gap-4">
             <button 
              onClick={fetchData} 
              className="bg-white text-gray-600 px-4 py-2 rounded-lg border hover:bg-gray-50 transition"
            >
              Refresh
            </button>
            <button 
              onClick={handleLogout} 
              className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
            >
              Logout
            </button>
          </div>
        </div>


        <MLControls />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h3 className="text-gray-500 text-sm font-medium">Total Users</h3>
            <p className="text-3xl font-bold text-blue-600">{users.length}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h3 className="text-gray-500 text-sm font-medium">Total Orders</h3>
            <p className="text-3xl font-bold text-green-600">
              {users.reduce((acc, user) => acc + user.orderCount, 0)}
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h3 className="text-gray-500 text-sm font-medium">Platform Activity</h3>
            <p className="text-3xl font-bold text-purple-600">Active</p>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <table className="w-full text-left">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="p-4 font-semibold text-gray-600">User / Profile</th>
                <th className="p-4 font-semibold text-gray-600">Joined</th>
                <th className="p-4 font-semibold text-gray-600 w-1/3">AI Recommendations (Score)</th>
                <th className="p-4 font-semibold text-gray-600">Stats</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {users.map((user) => (
                <tr key={user._id} className="hover:bg-gray-50 transition">
                  <td className="p-4 text-gray-800 font-medium leading-relaxed">
                    <div className="font-bold mb-1">{user.email}</div>
                    <div className="text-xs text-gray-500">
                        {user.age ? `Age: ${user.age}` : 'Age: ??'} • {user.gender || 'Gen: ??'}
                    </div>
                  </td>
                  <td className="p-4 text-gray-600">
                    {user.joinedAt ? new Date(user.joinedAt).toLocaleDateString() : 'N/A'}
                  </td>
                  <td className="p-4">
                     <div className="flex flex-col gap-1 max-h-40 overflow-auto">
                        {/* Top 4 Highlighted */}
                        {user.recommendations?.slice(0,4).map((rec, idx) => {
                             const scoreObj = user.scores?.find((s:any) => s.category === rec);
                             const score = scoreObj ? scoreObj.score : 'N/A';
                             return (
                                <div key={idx} className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded border border-green-100 flex justify-between">
                                    <span>{rec}</span>
                                    <span className="font-bold">{score}</span>
                                </div>
                             )
                        })}
                        {/* Others (Debug) */}
                         <details className="mt-2 group">
                            <summary className="text-xs text-blue-600 font-medium cursor-pointer hover:underline list-none flex items-center gap-1">
                               ▶ View All Candidates
                            </summary>
                            <div className="flex flex-col gap-1 mt-2 p-2 bg-gray-50 rounded border border-gray-100">
                                {user.scores && user.scores.length > 0 ? (
                                    user.scores
                                    .filter((s:any) => !user.recommendations?.slice(0,4).includes(s.category))
                                    .sort((a:any, b:any) => b.score - a.score)
                                    .map((s:any, idx:number) => (
                                     <div key={idx} className="flex flex-col border-b border-gray-100 last:border-0 py-1">
                                         <div className="flex justify-between text-xs text-gray-600">
                                            <span>{s.category}</span>
                                            <span className="font-mono text-gray-400">{s.score}</span>
                                         </div>
                                         <div className="text-[9px] text-gray-400 font-mono pl-1">
                                             {s.debug_details || ''}
                                         </div>
                                    </div>
                                    ))
                                ) : (
                                    <span className="text-[10px] text-gray-400 italic">
                                        No scores available. (Len: {user.scores ? user.scores.length : 'Undef'})
                                    </span>
                                )}
                            </div>
                        </details>
                     </div>
                  </td>
                  <td className="p-4 vertical-top">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      user.orderCount > 0 ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                    }`}>
                      {user.orderCount} Orders
                    </span>
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr>
                  <td colSpan={4} className="p-8 text-center text-gray-500">
                    No users found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
