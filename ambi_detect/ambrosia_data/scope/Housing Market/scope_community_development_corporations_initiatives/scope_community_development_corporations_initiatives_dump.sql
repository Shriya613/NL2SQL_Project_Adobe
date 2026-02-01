BEGIN TRANSACTION;
CREATE TABLE Community_Development_Corporations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Community_Development_Corporations" VALUES(1,'CDC1','New York');
INSERT INTO "Community_Development_Corporations" VALUES(2,'CDC2','Los Angeles');
INSERT INTO "Community_Development_Corporations" VALUES(3,'CDC3','Chicago');
INSERT INTO "Community_Development_Corporations" VALUES(4,'CDC4','Houston');
INSERT INTO "Community_Development_Corporations" VALUES(5,'CDC5','Philadelphia');
CREATE TABLE Community_Development_Corporations_Initiatives (
    community_development_corporation_id INTEGER,
    initiative_id INTEGER,
    FOREIGN KEY (community_development_corporation_id) REFERENCES Community_Development_Corporations(id),
    FOREIGN KEY (initiative_id) REFERENCES Initiatives(id)
);
INSERT INTO "Community_Development_Corporations_Initiatives" VALUES(1,1);
INSERT INTO "Community_Development_Corporations_Initiatives" VALUES(2,1);
INSERT INTO "Community_Development_Corporations_Initiatives" VALUES(3,1);
INSERT INTO "Community_Development_Corporations_Initiatives" VALUES(4,1);
INSERT INTO "Community_Development_Corporations_Initiatives" VALUES(5,1);
INSERT INTO "Community_Development_Corporations_Initiatives" VALUES(5,2);
CREATE TABLE Initiatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    initiative_name TEXT,
    description TEXT
);
INSERT INTO "Initiatives" VALUES(1,'Homeless_Shelter','A shelter providing housing for homeless individuals.');
INSERT INTO "Initiatives" VALUES(2,'Affordable_Housing','Building low-cost housing units for low-income families.');
INSERT INTO "Initiatives" VALUES(3,'Urban_Renewal','Revitalizing urban areas by improving infrastructure and public spaces.');
INSERT INTO "Initiatives" VALUES(4,'Eco_Friendly_Construction','Promoting sustainable construction practices.');
INSERT INTO "Initiatives" VALUES(5,'Youth_Empowerment','Programs aimed at empowering young people in the community.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Community_Development_Corporations',5);
INSERT INTO "sqlite_sequence" VALUES('Initiatives',5);
COMMIT;
