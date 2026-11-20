//! Exact prototype-cost adapter using the published ZKML chips.
//! See README.md for attribution, contract and adaptation boundaries.
use std::{collections::HashMap, fs::{self,File}, io::{BufReader,BufWriter,Write},
    path::Path, rc::Rc, sync::OnceLock, time::Instant};
use halo2_proofs::{circuit::{AssignedCell,Layouter,SimpleFloorPlanner,Value},
    dev::MockProver, halo2curves::{bn256::{Bn256,Fr,G1Affine},ff::{Field,PrimeField}},
    plonk::{Advice,Circuit,Column,ConstraintSystem,Error,Expression,Instance,
        ProvingKey,Selector,TableColumn,VerifyingKey,create_proof,keygen_pk,keygen_vk,verify_proof},
    poly::{commitment::Params,kzg::{commitment::{KZGCommitmentScheme,ParamsKZG},
        multiopen::{ProverSHPLONK,VerifierSHPLONK},strategy::SingleStrategy}},
    transcript::{Blake2bRead,Blake2bWrite,Challenge255,TranscriptReadBuffer,TranscriptWriterBuffer},
    SerdeFormat,poly::Rotation};
use halo2_gadgets::poseidon::primitives::{ConstantLength,Hash,Mds,Spec};
use serde_derive::{Deserialize,Serialize};
use serde_json::{json,Value as Json};
use sha2::{Digest,Sha256};
use zkml::{commitments::{commit::Commit,poseidon_commit::{PoseidonCommitChip,P128Pow5T3Gen}},
    gadgets::{adder::AdderChip,dot_prod::DotProductChip,gadget::{Gadget,GadgetConfig}}};

type Cell=AssignedCell<Fr,Fr>;
#[derive(Clone,Debug,Default,Serialize,Deserialize)]
struct Shape { n:usize,d:usize,centers:usize,k:u32,cols:usize }
#[derive(Clone,Debug)]
struct Config { g:Rc<GadgetConfig>, public:Column<Instance>, bytes:TableColumn,
    coord:Selector,cmp:Selector,min:Selector,dist:Selector,pack:Selector,
    hasher:PoseidonCommitChip<Fr,3,2,4> }
#[derive(Clone,Debug)]
struct Audit { shape:Shape,x:Vec<Vec<i64>>,centers:Vec<Vec<i64>>,salt:u64 }

fn field(v:i64)->Fr { if v<0 {-Fr::from(v.unsigned_abs())}else{Fr::from(v as u64)} }
fn known(c:&Cell)->Value<Fr> {c.value().copied()}
fn assign(region:&mut halo2_proofs::circuit::Region<'_,Fr>,col:Column<Advice>,row:usize,v:Value<Fr>)->Cell{
    region.assign_advice(||"value",col,row,||v).unwrap()
}
fn copy(region:&mut halo2_proofs::circuit::Region<'_,Fr>,col:Column<Advice>,row:usize,c:&Cell)->Cell{
    c.copy_advice(||"linked value",region,col,row).unwrap()
}
fn hash4(config:&Config,mut layouter:impl Layouter<Fr>,cells:[Cell;4])->Cell{
    let values=cells[..3].iter().cloned().map(Rc::new).collect();
    config.hasher.commit(layouter.namespace(||"upstream Poseidon"),config.g.clone(),
        &HashMap::new(),&values,Rc::new(cells[3].clone())).unwrap()[0].as_ref().clone()
}
impl Circuit<Fr> for Audit {
    type Config=Config;type FloorPlanner=SimpleFloorPlanner;type Params=Shape;
    fn params(&self)->Shape {self.shape.clone()}
    fn without_witnesses(&self)->Self {Self{shape:self.shape.clone(),
        x:vec![vec![0;self.shape.d];self.shape.n],
        centers:vec![vec![0;self.shape.d];self.shape.centers],salt:0}}
    fn configure(_: &mut ConstraintSystem<Fr>)->Config {panic!("shape parameters required")}
    fn configure_with_params(meta:&mut ConstraintSystem<Fr>,s:Shape)->Config{
        assert!(s.cols>=17 && s.cols<=33);
        let cols=(0..s.cols).map(|_|meta.advice_column()).collect::<Vec<_>>();
        for col in &cols {meta.enable_equality(*col);}
        let public=meta.instance_column();meta.enable_equality(public);
        let fixed=meta.fixed_column();meta.enable_constant(fixed);
        let g=GadgetConfig{columns:cols.clone(),fixed_columns:vec![fixed],
            num_cols:s.cols,use_selectors:true,..GadgetConfig::default()};
        let g=AdderChip::<Fr>::configure(meta,g);
        let g=DotProductChip::<Fr>::configure(meta,g);
        let coord=meta.complex_selector();let cmp=meta.complex_selector();
        let min=meta.selector();let dist=meta.selector();let pack=meta.selector();
        let bytes=meta.lookup_table_column();
        meta.create_gate("signed sixteen bit coordinate",|m|{
            let q=m.query_selector(coord);let a=m.query_advice(cols[0],Rotation::cur());
            let lo=m.query_advice(cols[1],Rotation::cur());let hi=m.query_advice(cols[2],Rotation::cur());
            vec![q*(a+Expression::Constant(Fr::from(32768))-lo-hi*Fr::from(256))]
        });
        for j in 1..=2 {meta.lookup("coordinate byte",|m|{
            let q=m.query_selector(coord);vec![(q*m.query_advice(cols[j],Rotation::cur()),bytes)]
        });}
        meta.create_gate("forty eight bit nonnegative difference",|m|{
            let q=m.query_selector(cmp);let a=m.query_advice(cols[0],Rotation::cur());
            let b=m.query_advice(cols[1],Rotation::cur());let mut v=Expression::Constant(Fr::ZERO);
            for j in 0..6 {v=v+m.query_advice(cols[j+2],Rotation::cur())*Fr::from(1u64<<(8*j));}
            vec![q*(a-b-v)]
        });
        for j in 2..8 {meta.lookup("comparison byte",|m|{
            let q=m.query_selector(cmp);vec![(q*m.query_advice(cols[j],Rotation::cur()),bytes)]
        });}
        meta.create_gate("exact minimum selection",|m|{
            let q=m.query_selector(min);let a=m.query_advice(cols[0],Rotation::cur());
            let b=m.query_advice(cols[1],Rotation::cur());let o=m.query_advice(cols[2],Rotation::cur());
            let pick=m.query_advice(cols[3],Rotation::cur());
            vec![q.clone()*pick.clone()*(pick.clone()-Expression::Constant(Fr::ONE)),
                q*(o-a.clone()-pick*(b-a))]
        });
        meta.create_gate("squared Euclidean distance",|m|{
            let q=m.query_selector(dist);let a=m.query_advice(cols[0],Rotation::cur());
            let b=m.query_advice(cols[1],Rotation::cur());let cross=m.query_advice(cols[2],Rotation::cur());
            let o=m.query_advice(cols[3],Rotation::cur());vec![q*(a+b-cross*Fr::from(2)-o)]
        });
        meta.create_gate("injective sixteen bit packing",|m|{
            let q=m.query_selector(pack);let mut acc=Expression::Constant(Fr::ZERO);let mut power=Fr::ONE;
            for col in cols.iter().take(15) {
                acc=acc+(m.query_advice(*col,Rotation::cur())+Expression::Constant(Fr::from(32768)))*power;
                power*=Fr::from(65536);
            }
            vec![q*(acc-m.query_advice(cols[15],Rotation::cur()))]
        });
        let state=[meta.advice_column(),meta.advice_column(),meta.advice_column()];
        for c in &state {meta.enable_equality(*c);}
        let partial=meta.advice_column();
        let inputs=[cols[0],cols[1],cols[2],cols[3]];
        let hasher=PoseidonCommitChip::<Fr,3,2,4>::configure(meta,inputs,state,partial);
        Config{g:Rc::new(g),public,bytes,coord,cmp,min,dist,pack,hasher}
    }
    fn synthesize(&self,c:Config,mut layouter:impl Layouter<Fr>)->Result<(),Error>{
        layouter.assign_table(||"byte table",|mut table|{
            for i in 0..256 {table.assign_cell(||"byte",c.bytes,i,||Value::known(Fr::from(i as u64)))?;}Ok(())
        })?;
        let cols=&c.g.columns;
        let (zero,header)=layouter.assign_region(||"schema and private salt",|mut r|{
            let zero=r.assign_advice_from_constant(||"zero",cols[0],0,Fr::ZERO)?;
            let domain=r.assign_advice_from_constant(||"domain",cols[0],1,Fr::from(2026102701))?;
            let n=r.assign_advice_from_constant(||"n",cols[0],2,Fr::from(self.shape.n as u64))?;
            let d=r.assign_advice_from_constant(||"d",cols[0],3,Fr::from(self.shape.d as u64))?;
            let salt=assign(&mut r,cols[0],4,Value::known(Fr::from(self.salt)));
            Ok((zero,[domain,n,d,salt]))
        })?;
        let all=self.x.iter().chain(self.centers.iter()).flat_map(|r|r.iter()).copied().collect::<Vec<_>>();
        let cells=layouter.assign_region(||"bounded original coordinates",|mut r|{
            let mut out=vec![];
            for (i,v) in all.iter().enumerate(){
                assert!((-32768..32768).contains(v));c.coord.enable(&mut r,i)?;
                out.push(assign(&mut r,cols[0],i,Value::known(field(*v))));
                let shifted=(v+32768) as u64;
                assign(&mut r,cols[1],i,Value::known(Fr::from(shifted&255)));
                assign(&mut r,cols[2],i,Value::known(Fr::from(shifted>>8)));
            }Ok(out)
        })?;
        let x_end=self.shape.n*self.shape.d;
        for (i,cell) in cells[x_end..].iter().enumerate(){layouter.constrain_instance(cell.cell(),c.public,i+1)?;}
        let mut root=hash4(&c,layouter.namespace(||"schema commitment"),header);
        let packed=layouter.assign_region(||"pack bounded catalog",|mut r|{
            let mut out=vec![];
            for (i,chunk) in cells[..x_end].chunks(15).enumerate(){
                c.pack.enable(&mut r,i)?;let mut value=Value::known(Fr::ZERO);let mut power=Fr::ONE;
                for j in 0..15 {
                    let cell=if j<chunk.len(){&chunk[j]}else{&zero};copy(&mut r,cols[j],i,cell);
                    value=value+(known(cell)+Value::known(Fr::from(32768)))*Value::known(power);
                    power*=Fr::from(65536);
                }out.push(assign(&mut r,cols[15],i,value));
            }Ok(out)
        })?;
        for chunk in packed.chunks(3){
            let a=chunk[0].clone();let b=chunk.get(1).unwrap_or(&zero).clone();let d=chunk.get(2).unwrap_or(&zero).clone();
            root=hash4(&c,layouter.namespace(||"catalog hash chain"),[root,a,b,d]);
        }
        layouter.constrain_instance(root.cell(),c.public,0)?;
        let dot=DotProductChip::<Fr>::construct(c.g.clone());let adder=AdderChip::<Fr>::construct(c.g.clone());
        let rows=cells[..x_end].chunks(self.shape.d).collect::<Vec<_>>();
        let centers=cells[x_end..].chunks(self.shape.d).collect::<Vec<_>>();
        let mut center_norms=vec![];
        for center in &centers{
            let refs=center.iter().collect::<Vec<_>>();
            center_norms.push(dot.forward(layouter.namespace(||"center norm"),&vec![refs.clone(),refs],&vec![&zero])?[0].clone());
        }
        let mut minima=vec![];
        for (i,row) in rows.iter().enumerate(){
            let refs=row.iter().collect::<Vec<_>>();
            let norm=dot.forward(layouter.namespace(||"row norm"),&vec![refs.clone(),refs.clone()],&vec![&zero])?[0].clone();
            let mut best:Option<(Cell,u64)>=None;
            for (j,center) in centers.iter().enumerate(){
                let cross=dot.forward(layouter.namespace(||"published exact dot product"),
                    &vec![refs.clone(),center.iter().collect()],&vec![&zero])?[0].clone();
                let exact=self.x[i].iter().zip(&self.centers[j]).map(|(a,b)|(*a-*b).pow(2) as u64).sum::<u64>();
                assert!(exact<(1u64<<48));
                let distance=layouter.assign_region(||"exact original-space distance",|mut r|{
                    c.dist.enable(&mut r,0)?;copy(&mut r,cols[0],0,&norm);copy(&mut r,cols[1],0,&center_norms[j]);
                    copy(&mut r,cols[2],0,&cross);Ok(assign(&mut r,cols[3],0,Value::known(Fr::from(exact))))
                })?;
                best=Some(if let Some((previous,previous_value))=best{
                    let pick=exact<previous_value;let value=previous_value.min(exact);
                    let selected=layouter.assign_region(||"select exact nearest center",|mut r|{
                        c.min.enable(&mut r,0)?;copy(&mut r,cols[0],0,&previous);copy(&mut r,cols[1],0,&distance);
                        let o=assign(&mut r,cols[2],0,Value::known(Fr::from(value)));
                        assign(&mut r,cols[3],0,Value::known(Fr::from(pick as u64)));Ok(o)
                    })?;
                    layouter.assign_region(||"prove both distance inequalities",|mut r|{
                        for (t,(cell,v)) in [(&previous,previous_value),(&distance,exact)].iter().enumerate(){
                            c.cmp.enable(&mut r,t)?;copy(&mut r,cols[0],t,cell);copy(&mut r,cols[1],t,&selected);
                            for b in 0..6{assign(&mut r,cols[b+2],t,Value::known(Fr::from((v-value)>>(8*b)&255)));}
                        }Ok(())
                    })?;(selected,value)
                }else{(distance,exact)});
            }minima.push(best.unwrap().0);
        }
        let total=adder.forward(layouter.namespace(||"complete catalog cost"),&vec![minima.iter().collect()],&vec![&zero])?;
        layouter.constrain_instance(total[0].cell(),c.public,1+self.shape.centers*self.shape.d)?;
        Ok(())
    }
}

#[derive(Debug)]
struct NativeSpec;
impl Spec<Fr,3,2> for NativeSpec{
    fn full_rounds()->usize{8} fn partial_rounds()->usize{56} fn secure_mds()->usize{0}
    fn sbox(v:Fr)->Fr{v.pow_vartime([5])}
    fn constants()->(Vec<[Fr;3]>,Mds<Fr,3>,Mds<Fr,3>){
        static CONSTANTS:OnceLock<(Vec<[Fr;3]>,Mds<Fr,3>,Mds<Fr,3>)>=OnceLock::new();
        CONSTANTS.get_or_init(||P128Pow5T3Gen::<Fr,0>::constants()).clone()
    }
}
fn h4(v:[Fr;4])->Fr{Hash::<Fr,NativeSpec,ConstantLength<4>,3,2>::init().hash(v)}
fn owner_commit(x:&[Vec<i64>],salt:u64)->Fr{
    let n=x.len();let d=x[0].len();let flat=x.iter().flatten().copied().collect::<Vec<_>>();
    let mut root=h4([Fr::from(2026102701),Fr::from(n as u64),Fr::from(d as u64),Fr::from(salt)]);
    let mut packed=vec![];
    for chunk in flat.chunks(15){let mut v=Fr::ZERO;let mut power=Fr::ONE;
        for j in 0..15{v+=Fr::from((chunk.get(j).copied().unwrap_or(0)+32768) as u64)*power;power*=Fr::from(65536);}packed.push(v);
    }
    for chunk in packed.chunks(3){root=h4([root,chunk[0],chunk.get(1).copied().unwrap_or(Fr::ZERO),chunk.get(2).copied().unwrap_or(Fr::ZERO)]);}root
}
fn hex(v:&[u8])->String{v.iter().map(|b|format!("{b:02x}")).collect()}
fn unhex(s:&str)->Vec<u8>{assert_eq!(s.len()%2,0);(0..s.len()).step_by(2).map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap()).collect()}
fn fields(values:&[String])->Vec<Fr>{values.iter().map(|s|{
    let v=unhex(s);let raw:<Fr as PrimeField>::Repr=v.try_into().unwrap();Option::<Fr>::from(Fr::from_repr(raw)).unwrap()
}).collect()}
fn read_json(p:impl AsRef<Path>)->Json{serde_json::from_reader(BufReader::new(File::open(p).unwrap())).unwrap()}
fn save(p:impl AsRef<Path>,v:&Json){fs::write(p,serde_json::to_vec_pretty(v).unwrap()).unwrap()}
fn digest_file(p:impl AsRef<Path>)->String{
    use std::io::Read;
    let mut file=File::open(p).unwrap();let mut hash=Sha256::new();let mut chunk=[0u8;65536];
    loop{let n=file.read(&mut chunk).unwrap();if n==0{break;}hash.update(&chunk[..n]);}hex(&hash.finalize())
}
fn matrix(v:&Json)->Vec<Vec<i64>>{serde_json::from_value(v.clone()).unwrap()}
fn input(path:&str,bank:usize,shape:Shape)->Audit{
    let v=read_json(path);let x=matrix(&v["x"]);let centers=matrix(&v["reuse_centers"][bank]);
    assert_eq!(x.len(),shape.n);assert_eq!(x[0].len(),shape.d);assert_eq!(centers.len(),shape.centers);
    assert!(x.iter().chain(centers.iter()).all(|r|r.len()==shape.d));
    Audit{shape,x,centers,salt:v["salt_x"].as_str().unwrap().parse().unwrap()}
}
fn statement(a:&Audit)->(Vec<Fr>,u64,Vec<usize>){
    let mut public=vec![owner_commit(&a.x,a.salt)];public.extend(a.centers.iter().flatten().map(|v|field(*v)));
    let mut labels=vec![];let mut total=0u64;
    for row in &a.x{let ds=a.centers.iter().map(|c|row.iter().zip(c).map(|(x,y)|(*x-*y).pow(2) as u64).sum::<u64>()).collect::<Vec<_>>();
        let label=(0..ds.len()).min_by_key(|j|ds[*j]).unwrap();labels.push(label);total+=ds[label];}
    public.push(Fr::from(total));(public,total,labels)
}
fn main(){
    let args=std::env::args().collect::<Vec<_>>();let mode=&args[1];
    if mode=="prepare" || mode=="inspect"{
        let source=&args[2];let out=Path::new(&args[3]);fs::create_dir_all(out).unwrap();
        let source_value=read_json(source);let xs=matrix(&source_value["x"]);let cs=matrix(&source_value["reuse_centers"][0]);
        let shape=Shape{n:xs.len(),d:xs[0].len(),centers:cs.len(),k:args[4].parse().unwrap(),cols:args[5].parse().unwrap()};
        assert!(shape.n<=4096 && shape.d<=10000 && shape.centers>=2 && shape.k<=23);
        let start=Instant::now();let circuit=input(source,0,shape.clone());let(public,_,_)=statement(&circuit);
        let owner_seconds=start.elapsed().as_secs_f64();
        if mode=="inspect"{
            let start=Instant::now();MockProver::run(shape.k,&circuit,vec![public]).unwrap().assert_satisfied();
            save(out.join("inspection.json"),&json!({"complete":true,"shape":shape,"mock_seconds":start.elapsed().as_secs_f64()}));
            println!("{}",json!({"complete":true,"phase":"inspect","shape":shape}));return;
        }
        // Match the published benchmark's local test parameter creation; retain its cost separately.
        let start=Instant::now();let params=ParamsKZG::<Bn256>::setup(shape.k,rand::rngs::OsRng);
        let mut writer=BufWriter::new(File::create(out.join("params")).unwrap());params.write(&mut writer).unwrap();writer.flush().unwrap();
        let params_seconds=start.elapsed().as_secs_f64();let start=Instant::now();
        let blank=circuit.without_witnesses();let vk=keygen_vk(&params,&blank).unwrap();let pk=keygen_pk(&params,vk,&blank).unwrap();
        let mut writer=BufWriter::new(File::create(out.join("pk")).unwrap());pk.write(&mut writer,SerdeFormat::RawBytes).unwrap();writer.flush().unwrap();
        fs::write(out.join("vk"),pk.get_vk().to_bytes(SerdeFormat::RawBytes)).unwrap();let setup_seconds=start.elapsed().as_secs_f64();
        let start=Instant::now();MockProver::run(shape.k,&circuit,vec![public.clone()]).unwrap().assert_satisfied();
        let mock_seconds=start.elapsed().as_secs_f64();
        let input_bits=source_value["model"]["input_bits"].as_u64().unwrap_or(12);
        save(out.join("owner_pin.json"),&json!({"shape":shape,"input_bits":input_bits,"catalog_commitment":hex(public[0].to_repr().as_ref()),
            "vk_sha256":digest_file(out.join("vk")),"params_sha256":digest_file(out.join("params")),"benchmark_parameters_only":true}));
        save(out.join("setup.json"),&json!({"complete":true,"owner_preparation_seconds":owner_seconds,"parameter_generation_seconds":params_seconds,"setup_seconds":setup_seconds,"mock_check_seconds":mock_seconds,"shape":shape,"pk_bytes":fs::metadata(out.join("pk")).unwrap().len(),"vk_bytes":fs::metadata(out.join("vk")).unwrap().len()}));
        println!("{}",json!({"complete":true,"phase":"prepare","shape":shape,"setup_seconds":setup_seconds}));
    }else if mode=="selftest"{
        let source=&args[2];let out=Path::new(&args[3]);let pin=read_json(out.join("owner_pin.json"));
        let shape:Shape=serde_json::from_value(pin["shape"].clone()).unwrap();let circuit=input(source,1,shape.clone());
        let(public,_,_)=statement(&circuit);
        let mut altered=circuit.clone();altered.x[0][0]+=1;
        assert!(MockProver::run(shape.k,&altered,vec![public.clone()]).unwrap().verify().is_err());
        altered=circuit.clone();altered.centers[0][0]+=1;
        assert!(MockProver::run(shape.k,&altered,vec![public.clone()]).unwrap().verify().is_err());
        let mut wrong_root=public.clone();wrong_root[0]+=Fr::ONE;
        assert!(MockProver::run(shape.k,&circuit,vec![wrong_root]).unwrap().verify().is_err());
        let mut wrong_cost=public;let last=wrong_cost.len()-1;wrong_cost[last]+=Fr::ONE;
        assert!(MockProver::run(shape.k,&circuit,vec![wrong_cost]).unwrap().verify().is_err());
        println!("{}",json!({"complete":true,"negative_witness_checks":4}));
    }else if mode=="query" || mode=="queries"{
        let source=&args[2];let out=Path::new(&args[3]);let whole=Instant::now();
        let banks=if mode=="query"{vec![args[4].parse::<usize>().unwrap()]}else{(0..read_json(source)["reuse_centers"].as_array().unwrap().len()).collect::<Vec<_>>()};
        let pin=read_json(out.join("owner_pin.json"));let shape:Shape=serde_json::from_value(pin["shape"].clone()).unwrap();
        let load=Instant::now();
        let params=ParamsKZG::<Bn256>::read(&mut BufReader::new(File::open(out.join("params")).unwrap())).unwrap();
        let pk=ProvingKey::<G1Affine>::read::<_,Audit>(&mut BufReader::new(File::open(out.join("pk")).unwrap()),SerdeFormat::RawBytes,shape.clone()).unwrap();
        let shared_load_seconds=load.elapsed().as_secs_f64();
        for (position,bank) in banks.iter().copied().enumerate(){
        let start=Instant::now();let circuit=input(source,bank,shape.clone());
        let(public,cost,labels)=statement(&circuit);assert_eq!(hex(public[0].to_repr().as_ref()),pin["catalog_commitment"].as_str().unwrap());
        let native_seconds=start.elapsed().as_secs_f64();let load_seconds=if position==0{shared_load_seconds}else{0.};let prove=Instant::now();
        let mut transcript=Blake2bWrite::<_,G1Affine,Challenge255<_>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<Bn256>,ProverSHPLONK<'_,Bn256>,Challenge255<G1Affine>,_,Blake2bWrite<Vec<u8>,G1Affine,Challenge255<_>>,Audit>(
            &params,&pk,&[circuit],&[&[&public]],rand::rngs::OsRng,&mut transcript).unwrap();
        let proof=transcript.finalize();let prove_seconds=prove.elapsed().as_secs_f64();
        let folder=out.join(format!("bank_{bank:03}"));fs::create_dir_all(&folder).unwrap();fs::write(folder.join("proof"),&proof).unwrap();
        save(folder.join("public.json"),&json!({"bank":bank,"public":public.iter().map(|v|hex(v.to_repr().as_ref())).collect::<Vec<_>>(),"cost":cost}));
        // Private diagnostics never enter the public verifier bundle.
        save(folder.join("private_diagnostics.json"),&json!({"labels":labels}));
        save(folder.join("result.json"),&json!({"complete":true,"bank":bank,"cost":cost,"native_seconds":native_seconds,"load_seconds":load_seconds,"prove_seconds":prove_seconds,"total_seconds":start.elapsed().as_secs_f64(),"proof_bytes":proof.len()}));
        println!("{}",json!({"complete":true,"bank":bank,"cost":cost,"prove_seconds":prove_seconds}));
        }
        save(out.join("queries.json"),&json!({"complete":true,"banks":banks,"shared_load_seconds":shared_load_seconds,"total_seconds":whole.elapsed().as_secs_f64()}));
    }else if mode=="verify"{
        // These directories contain approved pins/tasks and public proof material only.
        let out=Path::new(&args[2]);let bank:usize=args[3].parse().unwrap();let pin=read_json(&args[4]);let task=read_json(&args[5]);
        let start=Instant::now();let shape:Shape=serde_json::from_value(pin["shape"].clone()).unwrap();
        assert_eq!(digest_file(out.join("vk")),pin["vk_sha256"].as_str().unwrap());
        assert_eq!(digest_file(out.join("params")),pin["params_sha256"].as_str().unwrap());
        let statement=read_json(out.join(format!("bank_{bank:03}/public.json")));
        assert_eq!(statement["bank"].as_u64().unwrap() as usize,bank);
        let values:Vec<String>=serde_json::from_value(statement["public"].clone()).unwrap();let public=fields(&values);
        assert_eq!(values[0],pin["catalog_commitment"].as_str().unwrap());
        let centers=matrix(&task["reuse_centers"][bank]);assert_eq!(centers.len(),shape.centers);assert!(centers.iter().all(|r|r.len()==shape.d));
        let limit=1i64<<pin["input_bits"].as_u64().unwrap();assert!(centers.iter().flatten().all(|v|v.abs()<=limit));
        assert_eq!(public.len(),2+shape.centers*shape.d);
        assert!(public[1..public.len()-1].iter().zip(centers.iter().flatten()).all(|(a,b)|*a==field(*b)));
        let declared_cost=statement["cost"].as_u64().unwrap();assert_eq!(*public.last().unwrap(),Fr::from(declared_cost));
        let params=ParamsKZG::<Bn256>::read(&mut BufReader::new(File::open(out.join("params")).unwrap())).unwrap();
        let vk=VerifyingKey::<G1Affine>::read::<_,Audit>(&mut BufReader::new(File::open(out.join("vk")).unwrap()),SerdeFormat::RawBytes,shape).unwrap();
        let proof=fs::read(out.join(format!("bank_{bank:03}/proof"))).unwrap();
        let transcript=Blake2bRead::<_,G1Affine,Challenge255<_>>::init(&proof[..]);
        verify_proof::<KZGCommitmentScheme<Bn256>,VerifierSHPLONK<'_,Bn256>,Challenge255<G1Affine>,_,SingleStrategy<'_,Bn256>>(
            &params,&vk,SingleStrategy::new(&params),&[&[&public]],&mut {transcript}).unwrap();
        println!("{}",json!({"complete":true,"bank":bank,"cost":declared_cost,"verification_seconds":start.elapsed().as_secs_f64()}));
    }else{panic!("Expected prepare, query or verify");}
}
