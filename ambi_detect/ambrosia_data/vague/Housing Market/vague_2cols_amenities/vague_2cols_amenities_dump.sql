BEGIN TRANSACTION;
CREATE TABLE CondoBuildings (
    id INTEGER PRIMARY KEY,
    address TEXT,
    city TEXT,
    state TEXT,
    zipcode TEXT
);
INSERT INTO "CondoBuildings" VALUES(1,'123 Main St','New York','NY','10001');
INSERT INTO "CondoBuildings" VALUES(2,'456 Elm St','Chicago','IL','60601');
INSERT INTO "CondoBuildings" VALUES(3,'789 Oak St','Los Angeles','CA','90001');
INSERT INTO "CondoBuildings" VALUES(4,'246 Maple Ave','Houston','TX','77001');
INSERT INTO "CondoBuildings" VALUES(5,'135 Pine Blvd','Miami','FL','33101');
CREATE TABLE FacilitiesAndServices (
    id INTEGER PRIMARY KEY,
    facilities TEXT,
    services TEXT,
    condos_id INTEGER,
    FOREIGN KEY(condos_id) REFERENCES CondoBuildings(id)
);
INSERT INTO "FacilitiesAndServices" VALUES(1,'Pool','Gym',1);
INSERT INTO "FacilitiesAndServices" VALUES(2,'Sauna','Parking Lot',2);
INSERT INTO "FacilitiesAndServices" VALUES(3,'Laundry Room','Pet Friendly',3);
INSERT INTO "FacilitiesAndServices" VALUES(4,'Rooftop Deck','Security System',4);
INSERT INTO "FacilitiesAndServices" VALUES(5,'Playground','Storage Units',5);
CREATE TABLE Residents (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    building_id INTEGER,
    FOREIGN KEY(building_id) REFERENCES CondoBuildings(id)
);
INSERT INTO "Residents" VALUES(1,'John Doe',35,1);
INSERT INTO "Residents" VALUES(2,'Jane Smith',28,2);
INSERT INTO "Residents" VALUES(3,'Emma Jones',42,3);
INSERT INTO "Residents" VALUES(4,'Tom Brown',55,4);
INSERT INTO "Residents" VALUES(5,'Alice Johnson',30,5);
CREATE TABLE Transactions (
    id INTEGER PRIMARY KEY,
    resident_id INTEGER,
    purchase_price REAL,
    transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(resident_id) REFERENCES Residents(id)
);
INSERT INTO "Transactions" VALUES(1,1,250000.0,'2024-02-19 11:46:54');
INSERT INTO "Transactions" VALUES(2,2,300000.0,'2024-02-19 11:46:54');
INSERT INTO "Transactions" VALUES(3,3,450000.0,'2024-02-19 11:46:54');
INSERT INTO "Transactions" VALUES(4,4,550000.0,'2024-02-19 11:46:54');
INSERT INTO "Transactions" VALUES(5,5,650000.0,'2024-02-19 11:46:54');
COMMIT;
