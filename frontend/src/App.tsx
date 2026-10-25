import { Navigate, Route, Routes } from "react-router-dom";

import { Layout, Placeholder } from "./components/Layout";
import { DataPage } from "./pages/DataPage";
import { AnalyticsPage, ModelsPage, OverviewPage } from "./pages/InsightPages";
import { JobsPage } from "./pages/JobsPage";
import { LoginPage } from "./pages/LoginPage";
import { MediaDetailPage, MediaPage } from "./pages/MediaPages";
import { ProjectsPage } from "./pages/ProjectsPage";
import { ReviewPage } from "./pages/ReviewPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/projects" element={<ProjectsPage />} />
      <Route path="/p/:projectId" element={<Layout />}>
        <Route index element={<Navigate to="overview" replace />} />
        <Route path="overview" element={<OverviewPage />} />
        <Route path="data" element={<DataPage />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="media" element={<MediaPage />} />
        <Route path="media/:mediaId" element={<MediaDetailPage />} />
        <Route path="review" element={<ReviewPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="exports" element={<Placeholder title="Exports" week="W5–W6 (CSV, COCO; Camtrap DP in P1)" />} />
        <Route path="settings" element={<Placeholder title="Settings" week="W4 (users, α target, review budget, location precision)" />} />
      </Route>
      <Route path="*" element={<Navigate to="/projects" replace />} />
    </Routes>
  );
}
