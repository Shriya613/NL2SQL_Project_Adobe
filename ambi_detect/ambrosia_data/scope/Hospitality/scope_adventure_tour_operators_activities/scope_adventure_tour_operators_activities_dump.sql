BEGIN TRANSACTION;
CREATE TABLE Activities (
    id INTEGER PRIMARY KEY,
    activity_name TEXT,
    description TEXT
);
INSERT INTO "Activities" VALUES(1,'Zipline','An adrenaline-pumping aerial ride down a mountain.');
INSERT INTO "Activities" VALUES(2,'Rafting','Navigate white waters on an exciting river trip.');
INSERT INTO "Activities" VALUES(3,'Hiking','Explore scenic trails through beautiful landscapes.');
INSERT INTO "Activities" VALUES(4,'Biking','Cycle through picturesque countryside.');
INSERT INTO "Activities" VALUES(5,'Camping','Set up camp under starry skies in nature.');
CREATE TABLE Adventure_Tour_Operators (
    id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT
);
INSERT INTO "Adventure_Tour_Operators" VALUES(1,'Company A','New York');
INSERT INTO "Adventure_Tour_Operators" VALUES(2,'Company B','California');
INSERT INTO "Adventure_Tour_Operators" VALUES(3,'Company C','Texas');
INSERT INTO "Adventure_Tour_Operators" VALUES(4,'Company D','Florida');
INSERT INTO "Adventure_Tour_Operators" VALUES(5,'Company E','Arizona');
CREATE TABLE Adventure_Tour_Operators_Activities (
    adventure_tour_operator_id INTEGER,
    activity_id INTEGER,
    FOREIGN KEY(adventure_tour_operator_id) REFERENCES Adventure_Tour_Operators(id),
    FOREIGN KEY(activity_id) REFERENCES Activities(id)
);
INSERT INTO "Adventure_Tour_Operators_Activities" VALUES(1,1);
INSERT INTO "Adventure_Tour_Operators_Activities" VALUES(2,1);
INSERT INTO "Adventure_Tour_Operators_Activities" VALUES(3,1);
INSERT INTO "Adventure_Tour_Operators_Activities" VALUES(4,1);
INSERT INTO "Adventure_Tour_Operators_Activities" VALUES(5,1);
INSERT INTO "Adventure_Tour_Operators_Activities" VALUES(5,3);
COMMIT;
