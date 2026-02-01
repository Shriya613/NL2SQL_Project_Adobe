BEGIN TRANSACTION;
CREATE TABLE Real_Estate_Agencies (
    agency_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Real_Estate_Agencies" VALUES(1,'Agency A','New York');
INSERT INTO "Real_Estate_Agencies" VALUES(2,'Agency B','Los Angeles');
INSERT INTO "Real_Estate_Agencies" VALUES(3,'Agency C','Chicago');
INSERT INTO "Real_Estate_Agencies" VALUES(4,'Agency D','Miami');
INSERT INTO "Real_Estate_Agencies" VALUES(5,'Agency E','Dallas');
CREATE TABLE Real_Estate_Agencies_Services (
    real_estate_agency_id INTEGER,
    service_id INTEGER,
    FOREIGN KEY(real_estate_agency_id) REFERENCES Real_Estate_Agencies(agency_id),
    FOREIGN KEY(service_id) REFERENCES Services(service_id),
    PRIMARY KEY(real_estate_agency_id, service_id)
);
INSERT INTO "Real_Estate_Agencies_Services" VALUES(1,1);
INSERT INTO "Real_Estate_Agencies_Services" VALUES(2,1);
INSERT INTO "Real_Estate_Agencies_Services" VALUES(3,1);
INSERT INTO "Real_Estate_Agencies_Services" VALUES(4,1);
INSERT INTO "Real_Estate_Agencies_Services" VALUES(5,1);
INSERT INTO "Real_Estate_Agencies_Services" VALUES(5,2);
CREATE TABLE Services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT,
    price REAL
);
INSERT INTO "Services" VALUES(1,'Property Valuation',100.0);
INSERT INTO "Services" VALUES(2,'Rent Collection',200.0);
INSERT INTO "Services" VALUES(3,'Lease Preparation',300.0);
INSERT INTO "Services" VALUES(4,'Tenant Screening',400.0);
INSERT INTO "Services" VALUES(5,'Eviction Assistance',500.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Real_Estate_Agencies',5);
INSERT INTO "sqlite_sequence" VALUES('Services',5);
COMMIT;
