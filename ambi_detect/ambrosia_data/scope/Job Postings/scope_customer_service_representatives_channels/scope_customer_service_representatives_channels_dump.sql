BEGIN TRANSACTION;
CREATE TABLE Channels (
    channel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_value TEXT
);
INSERT INTO "Channels" VALUES(1,'Email');
INSERT INTO "Channels" VALUES(2,'Phone');
INSERT INTO "Channels" VALUES(3,'Chat');
INSERT INTO "Channels" VALUES(4,'In-person');
INSERT INTO "Channels" VALUES(5,'Social Media');
CREATE TABLE Customer_Service_Representatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT
, full_name TEXT);
INSERT INTO "Customer_Service_Representatives" VALUES(1,'john.doe@example.com','John Doe');
INSERT INTO "Customer_Service_Representatives" VALUES(2,'jane.smith@example.com','Jane Smith');
INSERT INTO "Customer_Service_Representatives" VALUES(3,'michael.johnson@example.com','Michael Johnson');
INSERT INTO "Customer_Service_Representatives" VALUES(4,'emily.williams@example.com','Emily Williams');
INSERT INTO "Customer_Service_Representatives" VALUES(5,'david.brown@example.com','David Brown');
CREATE TABLE Customer_Service_Representatives_Channels (
    representative_id INTEGER,
    channel_id INTEGER,
    FOREIGN KEY (representative_id) REFERENCES Customer_Service_Representatives(id),
    FOREIGN KEY (channel_id) REFERENCES Channels(channel_id)
);
INSERT INTO "Customer_Service_Representatives_Channels" VALUES(1,1);
INSERT INTO "Customer_Service_Representatives_Channels" VALUES(2,1);
INSERT INTO "Customer_Service_Representatives_Channels" VALUES(3,1);
INSERT INTO "Customer_Service_Representatives_Channels" VALUES(4,1);
INSERT INTO "Customer_Service_Representatives_Channels" VALUES(5,1);
INSERT INTO "Customer_Service_Representatives_Channels" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Customer_Service_Representatives',5);
INSERT INTO "sqlite_sequence" VALUES('Channels',5);
COMMIT;
