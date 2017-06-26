#include "framesdatabase.hpp"
using namespace litesql;
const litesql::FieldType FrameEntry::Own::Id("id_",A_field_type_integer,"FrameEntry_");
const std::string FrameEntry::type__("FrameEntry");
const std::string FrameEntry::table__("FrameEntry_");
const std::string FrameEntry::sequence__("FrameEntry_seq");
const litesql::FieldType FrameEntry::Id("id_",A_field_type_integer,table__);
const litesql::FieldType FrameEntry::Type("type_",A_field_type_string,table__);
const litesql::FieldType FrameEntry::Path("path_",A_field_type_string,table__);
const litesql::FieldType FrameEntry::Date("date_",A_field_type_datetime,table__);
const litesql::FieldType FrameEntry::Exposure("exposure_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Sharpness("sharpness_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Brightness("brightness_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Saturation("saturation_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Hue("hue_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Gain("gain_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Gamma("gamma_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Wbred("wbred_",A_field_type_float,table__);
const litesql::FieldType FrameEntry::Wbblue("wbblue_",A_field_type_float,table__);
void FrameEntry::initValues() {
}
void FrameEntry::defaults() {
    id = 0;
    date = 0;
    exposure = 0.0;
    sharpness = 0.0;
    brightness = 0.0;
    saturation = 0.0;
    hue = 0.0;
    gain = 0.0;
    gamma = 0.0;
    wbred = 0.0;
    wbblue = 0.0;
}
FrameEntry::FrameEntry(const litesql::Database& db)
     : litesql::Persistent(db), id(Id), type(Type), path(Path), date(Date), exposure(Exposure), sharpness(Sharpness), brightness(Brightness), saturation(Saturation), hue(Hue), gain(Gain), gamma(Gamma), wbred(Wbred), wbblue(Wbblue) {
    defaults();
}
FrameEntry::FrameEntry(const litesql::Database& db, const litesql::Record& rec)
     : litesql::Persistent(db, rec), id(Id), type(Type), path(Path), date(Date), exposure(Exposure), sharpness(Sharpness), brightness(Brightness), saturation(Saturation), hue(Hue), gain(Gain), gamma(Gamma), wbred(Wbred), wbblue(Wbblue) {
    defaults();
    size_t size = (rec.size() > 13) ? 13 : rec.size();
    switch(size) {
    case 13: wbblue = convert<const std::string&, float>(rec[12]);
        wbblue.setModified(false);
    case 12: wbred = convert<const std::string&, float>(rec[11]);
        wbred.setModified(false);
    case 11: gamma = convert<const std::string&, float>(rec[10]);
        gamma.setModified(false);
    case 10: gain = convert<const std::string&, float>(rec[9]);
        gain.setModified(false);
    case 9: hue = convert<const std::string&, float>(rec[8]);
        hue.setModified(false);
    case 8: saturation = convert<const std::string&, float>(rec[7]);
        saturation.setModified(false);
    case 7: brightness = convert<const std::string&, float>(rec[6]);
        brightness.setModified(false);
    case 6: sharpness = convert<const std::string&, float>(rec[5]);
        sharpness.setModified(false);
    case 5: exposure = convert<const std::string&, float>(rec[4]);
        exposure.setModified(false);
    case 4: date = convert<const std::string&, litesql::DateTime>(rec[3]);
        date.setModified(false);
    case 3: path = convert<const std::string&, std::string>(rec[2]);
        path.setModified(false);
    case 2: type = convert<const std::string&, std::string>(rec[1]);
        type.setModified(false);
    case 1: id = convert<const std::string&, int>(rec[0]);
        id.setModified(false);
    }
}
FrameEntry::FrameEntry(const FrameEntry& obj)
     : litesql::Persistent(obj), id(obj.id), type(obj.type), path(obj.path), date(obj.date), exposure(obj.exposure), sharpness(obj.sharpness), brightness(obj.brightness), saturation(obj.saturation), hue(obj.hue), gain(obj.gain), gamma(obj.gamma), wbred(obj.wbred), wbblue(obj.wbblue) {
}
const FrameEntry& FrameEntry::operator=(const FrameEntry& obj) {
    if (this != &obj) {
        id = obj.id;
        type = obj.type;
        path = obj.path;
        date = obj.date;
        exposure = obj.exposure;
        sharpness = obj.sharpness;
        brightness = obj.brightness;
        saturation = obj.saturation;
        hue = obj.hue;
        gain = obj.gain;
        gamma = obj.gamma;
        wbred = obj.wbred;
        wbblue = obj.wbblue;
    }
    litesql::Persistent::operator=(obj);
    return *this;
}
std::string FrameEntry::insert(litesql::Record& tables, litesql::Records& fieldRecs, litesql::Records& valueRecs) {
    tables.push_back(table__);
    litesql::Record fields;
    litesql::Record values;
    fields.push_back(id.name());
    values.push_back(id);
    id.setModified(false);
    fields.push_back(type.name());
    values.push_back(type);
    type.setModified(false);
    fields.push_back(path.name());
    values.push_back(path);
    path.setModified(false);
    fields.push_back(date.name());
    values.push_back(date);
    date.setModified(false);
    fields.push_back(exposure.name());
    values.push_back(exposure);
    exposure.setModified(false);
    fields.push_back(sharpness.name());
    values.push_back(sharpness);
    sharpness.setModified(false);
    fields.push_back(brightness.name());
    values.push_back(brightness);
    brightness.setModified(false);
    fields.push_back(saturation.name());
    values.push_back(saturation);
    saturation.setModified(false);
    fields.push_back(hue.name());
    values.push_back(hue);
    hue.setModified(false);
    fields.push_back(gain.name());
    values.push_back(gain);
    gain.setModified(false);
    fields.push_back(gamma.name());
    values.push_back(gamma);
    gamma.setModified(false);
    fields.push_back(wbred.name());
    values.push_back(wbred);
    wbred.setModified(false);
    fields.push_back(wbblue.name());
    values.push_back(wbblue);
    wbblue.setModified(false);
    fieldRecs.push_back(fields);
    valueRecs.push_back(values);
    return litesql::Persistent::insert(tables, fieldRecs, valueRecs, sequence__);
}
void FrameEntry::create() {
    litesql::Record tables;
    litesql::Records fieldRecs;
    litesql::Records valueRecs;
    type = type__;
    std::string newID = insert(tables, fieldRecs, valueRecs);
    if (id == 0)
        id = newID;
}
void FrameEntry::addUpdates(Updates& updates) {
    prepareUpdate(updates, table__);
    updateField(updates, table__, id);
    updateField(updates, table__, type);
    updateField(updates, table__, path);
    updateField(updates, table__, date);
    updateField(updates, table__, exposure);
    updateField(updates, table__, sharpness);
    updateField(updates, table__, brightness);
    updateField(updates, table__, saturation);
    updateField(updates, table__, hue);
    updateField(updates, table__, gain);
    updateField(updates, table__, gamma);
    updateField(updates, table__, wbred);
    updateField(updates, table__, wbblue);
}
void FrameEntry::addIDUpdates(Updates& ) {
}
void FrameEntry::getFieldTypes(std::vector<litesql::FieldType>& ftypes) {
    ftypes.push_back(Id);
    ftypes.push_back(Type);
    ftypes.push_back(Path);
    ftypes.push_back(Date);
    ftypes.push_back(Exposure);
    ftypes.push_back(Sharpness);
    ftypes.push_back(Brightness);
    ftypes.push_back(Saturation);
    ftypes.push_back(Hue);
    ftypes.push_back(Gain);
    ftypes.push_back(Gamma);
    ftypes.push_back(Wbred);
    ftypes.push_back(Wbblue);
}
void FrameEntry::delRecord() {
    deleteFromTable(table__, id);
}
void FrameEntry::delRelations() {
}
void FrameEntry::update() {
    if (!inDatabase) {
        create();
        return;
    }
    Updates updates;
    addUpdates(updates);
    if (id != oldKey) {
        if (!typeIsCorrect()) 
            upcastCopy()->addIDUpdates(updates);
    }
    litesql::Persistent::update(updates);
    oldKey = id;
}
void FrameEntry::del() {
    if (!typeIsCorrect()) {
        const std::unique_ptr<FrameEntry> p(upcastCopy());
        p->delRelations();
        p->onDelete();
        p->delRecord();
    } else {
        delRelations();
        onDelete();
        delRecord();
    }
    inDatabase = false;
}
bool FrameEntry::typeIsCorrect() const {
    return type == type__;
}
std::unique_ptr<FrameEntry> FrameEntry::upcast() const {
    return unique_ptr<FrameEntry>(new FrameEntry(*this));
}
std::unique_ptr<FrameEntry> FrameEntry::upcastCopy() const {
    FrameEntry* np = new FrameEntry(*this);
    np->id = id;
    np->type = type;
    np->path = path;
    np->date = date;
    np->exposure = exposure;
    np->sharpness = sharpness;
    np->brightness = brightness;
    np->saturation = saturation;
    np->hue = hue;
    np->gain = gain;
    np->gamma = gamma;
    np->wbred = wbred;
    np->wbblue = wbblue;
    np->inDatabase = inDatabase;
    return unique_ptr<FrameEntry>(np);
}
std::ostream & operator<<(std::ostream& os, FrameEntry const& o) {
    os << "-------------------------------------" << std::endl;
    os << o.id.name() << " = " << o.id << std::endl;
    os << o.type.name() << " = " << o.type << std::endl;
    os << o.path.name() << " = " << o.path << std::endl;
    os << o.date.name() << " = " << o.date << std::endl;
    os << o.exposure.name() << " = " << o.exposure << std::endl;
    os << o.sharpness.name() << " = " << o.sharpness << std::endl;
    os << o.brightness.name() << " = " << o.brightness << std::endl;
    os << o.saturation.name() << " = " << o.saturation << std::endl;
    os << o.hue.name() << " = " << o.hue << std::endl;
    os << o.gain.name() << " = " << o.gain << std::endl;
    os << o.gamma.name() << " = " << o.gamma << std::endl;
    os << o.wbred.name() << " = " << o.wbred << std::endl;
    os << o.wbblue.name() << " = " << o.wbblue << std::endl;
    os << "-------------------------------------" << std::endl;
    return os;
}
FramesDatabase::FramesDatabase(std::string backendType, std::string connInfo)
     : litesql::Database(backendType, connInfo) {
    initialize();
}
std::vector<litesql::Database::SchemaItem> FramesDatabase::getSchema() const {
    vector<Database::SchemaItem> res;
    string TEXT = backend->getSQLType(A_field_type_string);
    string rowIdType = backend->getRowIDType();
    res.push_back(Database::SchemaItem("schema_","table","CREATE TABLE schema_ (name_ "+TEXT+", type_ "+TEXT+", sql_ "+TEXT+")"));
    if (backend->supportsSequences()) {
        res.push_back(Database::SchemaItem("FrameEntry_seq","sequence",backend->getCreateSequenceSQL("FrameEntry_seq")));
    }
    res.push_back(Database::SchemaItem("FrameEntry_","table","CREATE TABLE FrameEntry_ (id_ " + rowIdType + ",type_ " + backend->getSQLType(A_field_type_string,"") + "" +",path_ " + backend->getSQLType(A_field_type_string,"") + "" +",date_ " + backend->getSQLType(A_field_type_datetime,"") + "" +",exposure_ " + backend->getSQLType(A_field_type_float,"") + "" +",sharpness_ " + backend->getSQLType(A_field_type_float,"") + "" +",brightness_ " + backend->getSQLType(A_field_type_float,"") + "" +",saturation_ " + backend->getSQLType(A_field_type_float,"") + "" +",hue_ " + backend->getSQLType(A_field_type_float,"") + "" +",gain_ " + backend->getSQLType(A_field_type_float,"") + "" +",gamma_ " + backend->getSQLType(A_field_type_float,"") + "" +",wbred_ " + backend->getSQLType(A_field_type_float,"") + "" +",wbblue_ " + backend->getSQLType(A_field_type_float,"") + "" +")"));
    res.push_back(Database::SchemaItem("FrameEntry_id_idx","index","CREATE INDEX FrameEntry_id_idx ON FrameEntry_ (id_)"));
    return res;
}
void FramesDatabase::initialize() {
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;
}
