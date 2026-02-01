BEGIN TRANSACTION;
CREATE TABLE Aircrafts (
    Aircraft_Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model_Name TEXT,
    Manufacturer TEXT,
    Registration_Mark TEXT,
    Type TEXT,
    Capacity INT,
    Range_Miles TEXT,
    Maximum_Speed MPH,
    Cruise_Speed MPH,
    Service_Ceiling FEET,
    Operator_Id INTEGER,
    FOREIGN KEY (Operator_Id) REFERENCES Airlines(Airline_Id)
);
INSERT INTO "Aircrafts" VALUES(1,'Boeing 787-8','Boeing','N123BA','Passenger',265,'7000',0.85,0.8,41000,1);
INSERT INTO "Aircrafts" VALUES(2,'Airbus A320neo','Airbus','F-HSEZ','Passenger',180,'3100',0.85,0.8,43000,2);
CREATE TABLE "Airports" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    IATA_Code CHAR(3),
    ICAA_Code CHAR(3),
    Latitude REAL,
    Longitude REAL,
    Elevation INT,
    Timezone INT,
    Direction_Indicator CHAR(2),
    Surface_Area FLOAT,
    Movement_Area FLOAT,
    Runway_Length FLOAT,
    Location TEXT,
    Phone_Number VARCHAR(15),
    Website VARCHAR(50),
    Email VARCHAR(50),
    Year_Established INT
);
INSERT INTO "Airports" VALUES(1,'John F Kennedy International Airport','JFK','N7988',-73.6954,-40.6452,13,-5,'R',1627.4,1627.4,3147.0,'New York','+1-800-222-1222','https://www.jfkairport.com/','info@jfkairport.com',1943);
INSERT INTO "Airports" VALUES(2,'Los Angeles International Airport','LAX','LACI',-118.4053,33.9435,279,-8,'W',2560.0,2560.0,9899.0,'Los Angeles','+1-855-463-5252','https://www.flylax.com/','information@flylax.com',1937);
CREATE TABLE Baggage (
    Bag_Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Flight_Id INTEGER,
    Owner_FirstName TEXT,
    Owner_LastName TEXT,
    Size_Inches TEXT,
    Weight_Kgs DOUBLE,
    Description TEXT,
    Status TEXT,
    FOREIGN KEY (Flight_Id) REFERENCES Flights(Flight_Id)
);
INSERT INTO "Baggage" VALUES(1,1,'John','Smith','22x15x10',10.0,'Business suit and laptop','Checked');
INSERT INTO "Baggage" VALUES(2,2,'Jane','Doe','21x14x9',8.0,'Personal items and toiletries','Carry on');
CREATE TABLE Flights (
    Flight_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Origin_Airport TEXT,
    Destination_Airport TEXT,
    Departure_Time TIMESTAMP,
    Arrival_Time TIMESTAMP,
    Carrier_Airlines TEXT,
    Aircraft_Type TEXT,
    Number_of_Passengers INT,
    Flight_Status TEXT,
    Gate_Number INT
);
INSERT INTO "Flights" VALUES(1,'JFK','LAX','2023-01-01 08:00:00','2023-01-01 11:00:00','American Airlines','Boeing 787 Dreamliner',250,'On Time',10);
INSERT INTO "Flights" VALUES(2,'Dallas/Fort Worth International Airport','Heathrow Airport','2023-01-02 14:00:00','2023-01-02 19:00:00','British Airways','Boeing 747 Jumbo Jet',300,'On Time',15);
CREATE TABLE Passengers (
    Passenger_Id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Date_Of_Birth DATE,
    Nationality TEXT,
    Passport_Number VARCHAR(15),
    Contact_Phone_Number VARCHAR(15),
    Seat_Preference TEXT,
    Special_Assistance BOOLEAN
);
INSERT INTO "Passengers" VALUES(1,'Alice','Johnson','1985-07-25','USA','US123456789','+1-555-123-4567','Window',0);
INSERT INTO "Passengers" VALUES(2,'Bob','Williams','1990-03-10','UK','GB987654321','+44-777-123-4567','Aisle',1);
CREATE TABLE Pilots (
    Pilot_Id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    License_Number VARCHAR(15),
    Date_Of_Birth DATE,
    Total_Flight_Hours INT,
    Rating TEXT,
    Current_Assignment TEXT,
    Hire_Date DATE
);
INSERT INTO "Pilots" VALUES(1,'John','Smith','ABCD1234','1985-06-15',5000,'ATP','Captain','2015-01-01');
INSERT INTO "Pilots" VALUES(2,'Jane','Doe','EFGH5678','1990-09-20',4000,'ATP','Co-pilot','2018-01-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Flights',2);
INSERT INTO "sqlite_sequence" VALUES('Pilots',2);
INSERT INTO "sqlite_sequence" VALUES('Passengers',2);
INSERT INTO "sqlite_sequence" VALUES('Baggage',2);
INSERT INTO "sqlite_sequence" VALUES('Aircrafts',2);
INSERT INTO "sqlite_sequence" VALUES('Airports',2);
COMMIT;
