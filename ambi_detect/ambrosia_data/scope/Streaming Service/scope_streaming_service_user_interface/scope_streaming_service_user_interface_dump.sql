BEGIN TRANSACTION;
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    price REAL
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix',12.99);
INSERT INTO "Streaming_Service" VALUES(2,'Amazon Prime Video',8.99);
INSERT INTO "Streaming_Service" VALUES(3,'Disney+',6.99);
INSERT INTO "Streaming_Service" VALUES(4,'Hulu',5.99);
INSERT INTO "Streaming_Service" VALUES(5,'HBO Max',14.99);
CREATE TABLE Streaming_Service_User_Interface (
    streaming_service_id INTEGER,
    user_interface_id INTEGER,
    FOREIGN KEY(streaming_service_id) REFERENCES Streaming_Service(id),
    FOREIGN KEY(user_interface_id) REFERENCES User_Interface(id)
);
INSERT INTO "Streaming_Service_User_Interface" VALUES(1,2);
INSERT INTO "Streaming_Service_User_Interface" VALUES(2,2);
INSERT INTO "Streaming_Service_User_Interface" VALUES(3,2);
INSERT INTO "Streaming_Service_User_Interface" VALUES(4,2);
INSERT INTO "Streaming_Service_User_Interface" VALUES(5,2);
INSERT INTO "Streaming_Service_User_Interface" VALUES(5,4);
CREATE TABLE User_Interface (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interface_type TEXT
);
INSERT INTO "User_Interface" VALUES(1,'Light Mode');
INSERT INTO "User_Interface" VALUES(2,'Dark Mode');
INSERT INTO "User_Interface" VALUES(3,'Classic Mode');
INSERT INTO "User_Interface" VALUES(4,'Minimalist Mode');
INSERT INTO "User_Interface" VALUES(5,'Customizable Mode');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('User_Interface',5);
COMMIT;
