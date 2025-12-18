export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="antialiased text-gray-900 bg-gray-50 min-h-screen">
       {/* Could add a Sidebar here later */}
      {children}
    </div>
  );
}
