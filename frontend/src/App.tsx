import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Delay from "./pages/Delay";
import Demand from "./pages/Demand";
import Products from "./pages/Products";
import Sellers from "./pages/Sellers";
import Reviews from "./pages/Reviews";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="retrasos" element={<Delay />} />
        <Route path="demanda" element={<Demand />} />
        <Route path="productos" element={<Products />} />
        <Route path="vendedores" element={<Sellers />} />
        <Route path="resenas" element={<Reviews />} />
      </Route>
    </Routes>
  );
}
