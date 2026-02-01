BEGIN TRANSACTION;
CREATE TABLE Bookings(
    Id INTEGER PRIMARY KEY,
    BookingId INTEGER,
    RoomNo INTEGER,
    CheckInTime TIME,
    CheckOutTime TIME,
    TotalStayDays INT,
    Price DECIMAL);
INSERT INTO "Bookings" VALUES(1,1,101,'10:00:00','14:00:00',2,150);
INSERT INTO "Bookings" VALUES(2,2,102,'11:00:00','15:00:00',2,170);
CREATE TABLE Customers(
    Id INTEGER PRIMARY KEY,
    CustomerID INTEGER,
    Fullname TEXT,
    Contact TEXT,
    ReservationDate DATETIME);
INSERT INTO "Customers" VALUES(1,1,'Alice Johnson','alice@example.com','2022-09-01T10:00:00');
INSERT INTO "Customers" VALUES(2,2,'Bob Smith','bob@example.com','2022-09-15T14:00:00');
CREATE TABLE Patios(
    Id INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    Phone TEXT,
    Number_of_Seats INT);
INSERT INTO "Patios" VALUES(1,'Sunny Side','123 Main St','555-6789',50);
INSERT INTO "Patios" VALUES(2,'Garden Oasis','456 Elm St','555-7890',60);
CREATE TABLE Rooms(
    Id INTEGER PRIMARY KEY,
    RoomNo INTEGER,
    Capacity INT,
    Amenities TEXT,
    FloorType TEXT);
INSERT INTO "Rooms" VALUES(1,101,8,'WiFi, TV, AC','Hardwood');
INSERT INTO "Rooms" VALUES(2,102,10,'WiFi, TV, AC','Tile');
CREATE TABLE Staffs(
    Id INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Position TEXT,
    Salary REAL);
INSERT INTO "Staffs" VALUES(1,'John','Smith','Manager',60000.0);
INSERT INTO "Staffs" VALUES(2,'Jane','Doe','Server',30000.0);
CREATE TABLE Verandas(
    Id INTEGER PRIMARY KEY,
    Name TEXT,
    Address TEXT,
    Phone TEXT,
    Number_of_Seats INT);
INSERT INTO "Verandas" VALUES(1,'Cozy Nook','789 Oak St','555-7654',50);
INSERT INTO "Verandas" VALUES(2,'Rooftop View','135 Pine St','555-6781',40);
COMMIT;
