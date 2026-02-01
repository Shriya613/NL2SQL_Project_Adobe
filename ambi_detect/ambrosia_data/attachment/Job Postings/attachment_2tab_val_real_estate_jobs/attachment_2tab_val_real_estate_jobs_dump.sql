BEGIN TRANSACTION;
CREATE TABLE Agent_Listings (
    agent_listing_id INTEGER PRIMARY KEY,
    real_estate_agent_id INTEGER,
    property_id INTEGER,
    FOREIGN KEY (real_estate_agent_id) REFERENCES Real_Estate_Agents(agent_id),
    FOREIGN KEY (property_id) REFERENCES Listed_Properties(property_id)
);
INSERT INTO "Agent_Listings" VALUES(1,1,1);
INSERT INTO "Agent_Listings" VALUES(2,2,2);
INSERT INTO "Agent_Listings" VALUES(3,3,3);
INSERT INTO "Agent_Listings" VALUES(4,1,4);
INSERT INTO "Agent_Listings" VALUES(5,2,5);
CREATE TABLE Clients (
    client_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    phone_number TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Clients" VALUES(1,'Tom','Hanks','+1112223333','tomhanks@email.com');
INSERT INTO "Clients" VALUES(2,'Emma','Watson','+4445556666','emmawatson@email.com');
INSERT INTO "Clients" VALUES(3,'Leonardo','DiCaprio','+7778889999','leodicaprio@email.com');
INSERT INTO "Clients" VALUES(4,'Kevin','Costner','+9998887777','kevincostner@email.com');
INSERT INTO "Clients" VALUES(5,'Angelina','Jolie','+10001001000','angelinajolie@email.com');
CREATE TABLE Listed_Properties (
    property_id INTEGER PRIMARY KEY,
    address TEXT,
    price REAL,
    listing_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT 
);
INSERT INTO "Listed_Properties" VALUES(1,'123 Main St','$500000','2024-03-13 17:23:06','Available');
INSERT INTO "Listed_Properties" VALUES(2,'456 Elm St','$300000','2024-03-13 17:23:06','Under Contract');
INSERT INTO "Listed_Properties" VALUES(3,'789 Oak St','$400000','2024-03-13 17:23:06','Active');
INSERT INTO "Listed_Properties" VALUES(4,'000 Pine St','$200000','2024-03-13 17:23:06','Pending Sale');
INSERT INTO "Listed_Properties" VALUES(5,'999 Fir St','$100000','2024-03-13 17:23:06','Rented');
CREATE TABLE Manager_Listings (
    manager_listing_id INTEGER PRIMARY KEY,
    property_manager_id INTEGER,
    property_id INTEGER,
    FOREIGN KEY (property_manager_id) REFERENCES Property_Managers(manager_id),
    FOREIGN KEY (property_id) REFERENCES Listed_Properties(property_id)
);
INSERT INTO "Manager_Listings" VALUES(1,1,1);
INSERT INTO "Manager_Listings" VALUES(2,2,2);
INSERT INTO "Manager_Listings" VALUES(3,3,3);
INSERT INTO "Manager_Listings" VALUES(4,1,4);
INSERT INTO "Manager_Listings" VALUES(5,2,5);
CREATE TABLE Property_Managers (
    manager_id INTEGER PRIMARY KEY,
    name TEXT,
    phone_number TEXT,
    email TEXT,
    commission VARCHAR(10) 
);
INSERT INTO "Property_Managers" VALUES(1,'Alice Brown','+9876543211','alice@example.com','5%');
INSERT INTO "Property_Managers" VALUES(2,'Bob White','+1234567891','bob@example.com','3%');
INSERT INTO "Property_Managers" VALUES(3,'Charlie Black','+1122334455','charlie@example.com','2%');
CREATE TABLE Real_Estate_Agents (
    agent_id INTEGER PRIMARY KEY,
    name TEXT,
    phone_number TEXT,
    email TEXT,
    commission VARCHAR(10) 
);
INSERT INTO "Real_Estate_Agents" VALUES(1,'John Doe','+1234567890','john@example.com','5%');
INSERT INTO "Real_Estate_Agents" VALUES(2,'Jane Smith','+0987654321','jane@example.com','3%');
INSERT INTO "Real_Estate_Agents" VALUES(3,'Mike Johnson','+9876543210','mike@example.com','2%');
COMMIT;
