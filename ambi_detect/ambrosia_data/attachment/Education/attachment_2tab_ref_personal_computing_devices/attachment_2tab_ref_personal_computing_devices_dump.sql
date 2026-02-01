BEGIN TRANSACTION;
CREATE TABLE Laptops(
    Laptop_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Processor TEXT,
    RAM_Size INT,
    Hard_Disk_Space INT,
    Graphics_Card TEXT,
    Weight REAL,
    Price REAL,
    OS_ID INTEGER,
    FOREIGN KEY (OS_ID) REFERENCES Operating_System(OS_ID));
INSERT INTO "Laptops" VALUES(1,'XPS 15','Intel Core i7-10875H',32,1024,'NVIDIA GeForce GTX 1650 Ti',4.5,1999.99,1);
INSERT INTO "Laptops" VALUES(2,'MacBook Pro 16','Intel Core i7-9750H',16,512,'AMD Radeon Pro 5300M',4.3,2399.99,2);
INSERT INTO "Laptops" VALUES(3,'ThinkPad T490s','Intel Core i7-10610U',16,1024,'Intel UHD Graphics',3.1,1499.99,3);
INSERT INTO "Laptops" VALUES(4,'ZenBook Pro Duo','Intel Core i9-10980HK',32,1024,'NVIDIA GeForce RTX 2060',4.5,2599.99,4);
INSERT INTO "Laptops" VALUES(5,'Yoga C940','Intel Core i7-10750H',16,1024,'NVIDIA GeForce MX350',3.7,1299.99,5);
CREATE TABLE Operating_System(
    OS_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT);
INSERT INTO "Operating_System" VALUES(1,'Windows 10');
INSERT INTO "Operating_System" VALUES(2,'macOS Big Sur');
INSERT INTO "Operating_System" VALUES(3,'Ubuntu 20.04 LTS');
INSERT INTO "Operating_System" VALUES(4,'Android 11');
INSERT INTO "Operating_System" VALUES(5,'iOS 14');
CREATE TABLE Students(
    Student_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT,
    Last_Name TEXT,
    Age INT,
    Grade_Level TEXT,
    Tablet_ID INTEGER,
    Laptop_ID INTEGER,
    FOREIGN KEY (Tablet_ID) REFERENCES Tablets(Tablet_ID),
    FOREIGN KEY (Laptop_ID) REFERENCES Laptops(Laptop_ID));
INSERT INTO "Students" VALUES(1,'John','Smith',15,'10th',1,3);
INSERT INTO "Students" VALUES(2,'Jane','Doe',14,'9th',2,4);
INSERT INTO "Students" VALUES(3,'Alice','Brown',13,'8th',3,2);
INSERT INTO "Students" VALUES(4,'Tom','Williams',15,'10th',4,5);
INSERT INTO "Students" VALUES(5,'Emma','Jackson',14,'9th',5,1);
CREATE TABLE Tablets(
    Tablet_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Model TEXT,
    Screen_Size REAL,
    RAM_Size INT,
    Storage_Capacity INT,
    Purchase_Price REAL,
    OS_ID INTEGER,
    FOREIGN KEY (OS_ID) REFERENCES Operating_System(OS_ID));
INSERT INTO "Tablets" VALUES(1,'Surface Pro X',13.0,8,256,999.99,1);
INSERT INTO "Tablets" VALUES(2,'iPad Air',10.9,4,64,599.99,2);
INSERT INTO "Tablets" VALUES(3,'Galaxy Tab S7+',12.4,6,128,849.99,3);
INSERT INTO "Tablets" VALUES(4,'Pixel Slate',12.3,8,64,599.99,4);
INSERT INTO "Tablets" VALUES(5,'Lenovo Yoga C940',14.0,16,512,1099.99,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Operating_System',5);
INSERT INTO "sqlite_sequence" VALUES('Tablets',5);
INSERT INTO "sqlite_sequence" VALUES('Laptops',5);
INSERT INTO "sqlite_sequence" VALUES('Students',5);
COMMIT;
