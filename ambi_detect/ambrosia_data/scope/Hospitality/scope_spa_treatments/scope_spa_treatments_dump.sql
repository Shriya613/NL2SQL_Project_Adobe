BEGIN TRANSACTION;
CREATE TABLE Appointments (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    customer_id INTEGER,
    spa_id INTEGER,
    treatment_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id),
    FOREIGN KEY (spa_id) REFERENCES Spa(spa_id),
    FOREIGN KEY (treatment_id) REFERENCES Treatments(treatment_id)
);
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT
);
INSERT INTO "Customers" VALUES(1,'Jane','Doe','jane.doe@example.com');
INSERT INTO "Customers" VALUES(2,'John','Smith','john.smith@example.com');
INSERT INTO "Customers" VALUES(3,'Alice','Johnson','alice.johnson@example.com');
INSERT INTO "Customers" VALUES(4,'Bob','Williams','bob.williams@example.com');
INSERT INTO "Customers" VALUES(5,'Emily','Davis','emily.davis@example.com');
CREATE TABLE Spa (
    spa_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    phone_number TEXT
);
INSERT INTO "Spa" VALUES(1,'Spalicious','123 Main St','555-123-4567');
INSERT INTO "Spa" VALUES(2,'The Serene Oasis','456 Elm St','555-987-6543');
INSERT INTO "Spa" VALUES(3,'Aromatherapy Haven','789 Maple Ave','555-321-0987');
INSERT INTO "Spa" VALUES(4,'Tranquil Retreat','987 Oak St','555-456-7890');
INSERT INTO "Spa" VALUES(5,'Rejuvenation Sanctuary','135 Pine St','555-234-5678');
CREATE TABLE Spa_Treatments (
    spa_treatment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    spa_id INTEGER,
    treatment_id INTEGER,
    FOREIGN KEY (spa_id) REFERENCES Spa(spa_id),
    FOREIGN KEY (treatment_id) REFERENCES Treatments(treatment_id)
);
INSERT INTO "Spa_Treatments" VALUES(1,1,1);
INSERT INTO "Spa_Treatments" VALUES(2,2,1);
INSERT INTO "Spa_Treatments" VALUES(3,3,1);
INSERT INTO "Spa_Treatments" VALUES(4,4,1);
INSERT INTO "Spa_Treatments" VALUES(5,5,1);
INSERT INTO "Spa_Treatments" VALUES(6,5,2);
CREATE TABLE Treatments (
    treatment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Treatments" VALUES(1,'Massage','Relaxing Swedish massage with essential oils.');
INSERT INTO "Treatments" VALUES(2,'Facial','Revitalizing facial treatment using natural products.');
INSERT INTO "Treatments" VALUES(3,'Manicure','Nourishing manicure with nail polish application.');
INSERT INTO "Treatments" VALUES(4,'Pedicure','Moisturizing pedicure with callus removal.');
INSERT INTO "Treatments" VALUES(5,'Body Scrub','Exfoliating body scrub with sea salt and oil.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Spa',5);
INSERT INTO "sqlite_sequence" VALUES('Treatments',5);
INSERT INTO "sqlite_sequence" VALUES('Customers',5);
INSERT INTO "sqlite_sequence" VALUES('Spa_Treatments',6);
COMMIT;
