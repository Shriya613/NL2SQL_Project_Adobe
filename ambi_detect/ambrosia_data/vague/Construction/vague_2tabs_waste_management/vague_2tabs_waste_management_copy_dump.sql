BEGIN TRANSACTION;
CREATE TABLE Construction_Material (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_type TEXT,
    quantity REAL
);
INSERT INTO "Construction_Material" VALUES(1,'Concrete',100.0);
INSERT INTO "Construction_Material" VALUES(2,'Steel',200.0);
INSERT INTO "Construction_Material" VALUES(3,'Bricks',300.0);
INSERT INTO "Construction_Material" VALUES(4,'Glass',400.0);
INSERT INTO "Construction_Material" VALUES(5,'Aluminum',500.0);
CREATE TABLE "Construction_Site" (
    site_id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_name TEXT,
    location TEXT,
    waste_disposal_method_id INTEGER,
    FOREIGN KEY(waste_disposal_method_id) REFERENCES "Waste_Management"(id));
INSERT INTO "Construction_Site" VALUES(1,'Site A','New York City',1);
INSERT INTO "Construction_Site" VALUES(2,'Site B','Los Angeles',2);
INSERT INTO "Construction_Site" VALUES(3,'Site C','Chicago',3);
INSERT INTO "Construction_Site" VALUES(4,'Site D','Miami',4);
INSERT INTO "Construction_Site" VALUES(5,'Site E','Dallas',5);
CREATE TABLE "Disposed_Waste" (
    disposed_waste_id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_material_id INTEGER,
    waste_disposal_method_id INTEGER,
    disposal_date DATE,
    FOREIGN KEY(site_material_id) REFERENCES Site_Material(site_material_id),
    FOREIGN KEY(waste_disposal_method_id) REFERENCES "Waste_Management"(id)
);
INSERT INTO "Disposed_Waste" VALUES(1,1,1,'2022-01-01');
INSERT INTO "Disposed_Waste" VALUES(2,1,1,'2022-01-02');
INSERT INTO "Disposed_Waste" VALUES(3,2,2,'2022-01-03');
INSERT INTO "Disposed_Waste" VALUES(4,2,2,'2022-01-04');
INSERT INTO "Disposed_Waste" VALUES(5,3,3,'2022-01-05');
CREATE TABLE Site_Material (
    site_material_id INTEGER PRIMARY KEY AUTOINCREMENT,
    construction_site_id INTEGER,
    construction_material_id INTEGER,
    amount REAL,
    FOREIGN KEY(construction_site_id) REFERENCES Construction_Site(site_id),
    FOREIGN KEY(construction_material_id) REFERENCES Construction_Material(id)
);
INSERT INTO "Site_Material" VALUES(1,1,1,50.0);
INSERT INTO "Site_Material" VALUES(2,1,2,75.0);
INSERT INTO "Site_Material" VALUES(3,2,3,60.0);
INSERT INTO "Site_Material" VALUES(4,2,4,80.0);
INSERT INTO "Site_Material" VALUES(5,3,5,90.0);
CREATE TABLE "Waste_Management" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    method_name TEXT,
    description TEXT
);
INSERT INTO "Waste_Management" VALUES(1,'Landfill','Disposing of waste by burying it in a designated area.');
INSERT INTO "Waste_Management" VALUES(2,'Incineration','Burning waste to reduce its volume and kill pathogens.');
INSERT INTO "Waste_Management" VALUES(3,'Recycling','Processing waste materials to create new products.');
INSERT INTO "Waste_Management" VALUES(4,'Composting','Breaking down organic matter into nutrient-rich humus.');
INSERT INTO "Waste_Management" VALUES(5,'Biodegradable Waste','Organic waste that can be broken down by microorganisms.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Waste_Management',5);
INSERT INTO "sqlite_sequence" VALUES('Construction_Material',5);
INSERT INTO "sqlite_sequence" VALUES('Site_Material',5);
INSERT INTO "sqlite_sequence" VALUES('Construction_Site',5);
INSERT INTO "sqlite_sequence" VALUES('Disposed_Waste',5);
COMMIT;
