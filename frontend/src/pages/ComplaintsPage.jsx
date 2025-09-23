import React from 'react';
import ComplaintForm from '../components/ComplaintForm';
import MapView from '../components/MapView';
import { useTranslation } from "react-i18next";
import Header from '../components/Header';

function ComplaintsPage() {
    const { t } = useTranslation();

    return (
        <div >
            <Header name={t("crowdsourced_farm_reporting")} />
            <ComplaintForm />
            <hr />
            <MapView />
        </div>
    );
}

export default ComplaintsPage;