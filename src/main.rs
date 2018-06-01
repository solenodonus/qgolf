extern crate gif;
extern crate num;

#[allow(non_camel_case_types)]
type real = f64;

#[allow(non_camel_case_types)]
type complex = num::complex::Complex<real>;

#[derive(Clone)]
struct Field<T: Copy> {
  width: usize,
  height: usize,
  data: Vec<T>,
}

impl <T: Copy> Field<T> {
  fn assert_position_ok(&self, x: usize, y: usize) {
    assert!(x < self.width && y < self.height,
            "Field position out of range: ({}, {}) vs ({}, {})",
            &x, &y, &self.width, &self.height);
  }

  fn at(&self, x: usize, y: usize) -> T {
    self.assert_position_ok(x, y);
    self.data[x * self.height + y]
  }

  fn at_mut(&mut self, x: usize, y: usize) -> &mut T {
    self.assert_position_ok(x, y);
    &mut self.data[x * self.height + y]
  }

  fn set(&mut self, x: usize, y: usize, t: T) {
    *self.at_mut(x, y) = t;
  }
}

fn apply_boundary_conditions(psi: &mut Field<complex>) {
  // Dirichlet boundary conditions.
  for x in 0 .. psi.width {
    for y in 0 .. psi.height {
      if x <= 1 || x >= psi.width - 2 || y <= 1 || y >= psi.height - 2 {
        psi.set(x, y, complex::new(0.0, 0.0));
      }
    }
  }
}

impl <T: Copy + Default> Field<T> {
  fn new(width: usize, height: usize) -> Field<T> {
    Field {
      width, height,
      data: vec![T::default(); width * height],
    }
  }

  fn new_populated<F>(width: usize, height: usize, pop_f: F) -> Field<T>
      where F: Fn(usize, usize) -> T {
    let mut res = Field::<T>::new(width, height);
    for x in 0 .. width {
      for y in 0 .. height {
        res.set(x, y, pop_f(x, y));
      }
    }
    return res;
  }
}

fn wave_function_normalization(psi: &Field<complex>) -> real {
  let mut norm_sqr: real = 0.0;
  for e in &psi.data {
    norm_sqr += e.norm_sqr();
  }
  return norm_sqr.sqrt();
}

fn normalize_wave_function(psi: &mut Field<complex>) {
  let norm = wave_function_normalization(psi);
  for e in psi.data.iter_mut() {
    *e /= norm;
  }
}

fn laplacian(psi: &Field<complex>, x: usize, y: usize) -> complex {
  psi.assert_position_ok(x, y);
  assert!(x > 1 && x < psi.width - 2 && y > 1 && y < psi.height - 2,
          "Laplacian cannot be calculated at boundary: ({}, {})", &x, &y);
  // Fourth-order approximated lattice laplacian.
  - 3.0 * psi.at(x, y)
  + 0.5 * (psi.at(x + 1, y) + psi.at(x - 1, y) +
           psi.at(x, y + 1) + psi.at(x, y - 1))
  + (4.0 / 9.0) * (psi.at(x + 1, y + 1) + psi.at(x + 1, y - 1) +
                   psi.at(x - 1, y + 1) + psi.at(x - 1, y - 1))
  + (3.0 / 8.0) * (psi.at(x + 2, y) + psi.at(x, y + 2) +
                   psi.at(x - 2, y) + psi.at(x, y - 2))
  - (13.0 / 36.0) * (psi.at(x + 2, y + 1) + psi.at(x + 1, y + 2) +
                     psi.at(x + 2, y - 1) + psi.at(x + 1, y - 2) +
                     psi.at(x - 2, y + 1) + psi.at(x - 1, y + 2) +
                     psi.at(x - 2, y - 1) + psi.at(x - 1, y - 2))
  + (11.0 / 72.0) * (psi.at(x + 2, y + 2) + psi.at(x + 2, y - 2) +
                     psi.at(x - 2, y + 2) + psi.at(x - 2, y - 2))
}

fn compute_diff(psi: &Field<complex>, u: &Field<real>,
                dt: real, hbar: real) -> Field<complex> {
  let mut res = Field::<complex>::new(psi.width, psi.height);
  for x in 2 .. psi.width - 2 {
    for y in 2 .. psi.height - 2 {
      res.set(x, y, (complex::new(0.0, 0.5 * hbar) * laplacian(psi, x, y) -
                     complex::new(0.0, 1.0 / hbar) * u.at(x, y) * psi.at(x, y))
              * dt);
    }
  }
  return res;
}

fn evolve_single_step(psi: &mut Field<complex>, u: &Field<real>,
                          dt: real, hbar: real) {
  let k1 = compute_diff(psi, u, dt, hbar);

  let k2_arg = Field::<complex>::new_populated(
      psi.width, psi.height, |x, y| psi.at(x, y) + k1.at(x, y) / 2.0);
  let k2 = compute_diff(&k2_arg, u, dt, hbar);

  let k3_arg = Field::<complex>::new_populated(
      psi.width, psi.height, |x, y| psi.at(x, y) + k2.at(x, y) / 2.0);
  let k3 = compute_diff(&k3_arg, u, dt, hbar);

  let k4_arg = Field::<complex>::new_populated(
      psi.width, psi.height, |x, y| psi.at(x, y) + k3.at(x, y));
  let k4 = compute_diff(&k4_arg, u, dt, hbar);

  for x in 0 .. psi.width {
    for y in 0 .. psi.height {
      let old_psi = psi.at(x, y);
      psi.set(x, y, old_psi + (k1.at(x, y) + k2.at(x, y) * 2.0 +
              k3.at(x, y) * 2.0 + k4.at(x, y)) / 6.0);
    }
  }
}

fn evolve_fixed_steps(psi: &mut Field<complex>, u: &Field<real>,
                      dt: real, hbar: real, n: usize) {
  let dt_effective = dt / n as real;
  for _ in 0..n {
    evolve_single_step(psi, u, dt_effective, hbar);
  }
}

fn evolve(psi: &mut Field<complex>, u: &Field<real>, dt: real,
          hbar: real, mut n: usize) {
  let psi_copy = psi.clone();
  let mut last_normalization: real = -1.0;
  loop {
    print!("Trying to evolve in {} steps.. ", &n);
    evolve_fixed_steps(psi, u, dt, hbar, n);
    let normalization = wave_function_normalization(psi);
    if last_normalization != -1.0 &&
       (normalization - 1.0).abs() > (last_normalization - 1.0).abs() {
      println!("settling for normalization {}", &normalization);
      break;
    } else if normalization < 0.99 || normalization > 1.01 {
      println!("failure: normalization {}", &normalization);
      *psi = psi_copy.clone();
      n *= 2;
      last_normalization = normalization
    } else {
      println!("success, normalization {}", &normalization);
      break;
    }
  }
  normalize_wave_function(psi);
}

fn displayed_intensity(wf: complex) -> real {
  wf.norm_sqr()
}

struct RenderOpts {
  shades_num: u8,             // Nuber of levels of color.
}

fn generate_palette(opts: &RenderOpts) -> Vec<u8> {
  let (r, g, b) = (0xFF, 0xFF, 0xFF);
  let mut palette = Vec::<u8>::with_capacity(opts.shades_num as usize * 3);
  for i in 0 .. opts.shades_num {
    palette.push(((r as u32) * (i as u32) / (opts.shades_num as u32)) as u8);
    palette.push(((g as u32) * (i as u32) / (opts.shades_num as u32)) as u8);
    palette.push(((b as u32) * (i as u32) / (opts.shades_num as u32)) as u8);
  }
  return palette;
}

fn render_frame(psi: &Field<complex>, buffer: &mut [u8], opts: &RenderOpts) {
  let mut color_normalization = 0.0;
  for x in 0 .. psi.width {
    for y in 0 .. psi.height {
      let intensity = displayed_intensity(psi.at(x, y));
      if intensity > color_normalization {
        color_normalization = intensity;
      }
    }
  }

  assert!(buffer.len() == psi.width * psi.height,
          "Buffer size mismatch: {} vs {}", buffer.len(),
          psi.width * psi.height);
  for x in 0 .. psi.width {
    for y in 0 .. psi.height {
      let intensity = displayed_intensity(psi.at(x, y));
      let relative_intensity = intensity / color_normalization;
      let mut shade = (relative_intensity * opts.shades_num as f64)
          .floor() as u8;
      if shade == opts.shades_num {
        shade -= 1;
      }
      buffer[y * psi.width + x] = shade;
    }
  }
}

fn main() {
  let (w, h) = (640, 480);  // Resolution of the GIF.
  let mut psi = Field::<complex>::new(w, h);  // Wave function.

  let hbar: real = 0.1;  // Reduced Planck's constant (aka h-bar), in natural units.
  let (x0, y0) = (320, 240);  // Initial position of the golf ball.
  let (p0_x, p0_y): (real, real) =
                    (100.0, 30.0);  // Initial momenta of the golf ball.

  let lattice_spacing: real = 0.01;

  let frames_count = 300;
  let dt: real = 50.0;

  let steps_per_frame = 20;
  let individual_frames = false;

  // 0. Prepare the potential energy function (initial conditions).
  let stiffness = 0.000;
  let mut u = Field::<real>::new(w, h);
  for x in 0..w {
    for y in 0..h {
      u.set(x, y, (((x as real - x0 as real) * lattice_spacing).powi(4) +
                   ((y as real - y0 as real) * lattice_spacing).powi(4)) *
                   stiffness);
    }
  }

  // 1. Initialize the wavefunction with a coherent state.
  for x in 0..w {
    for y in 0..h {
      let square_arg_part = complex::new(
        - (((x as real - x0 as real) * lattice_spacing).powi(2) +
           ((y as real - y0 as real) * lattice_spacing).powi(2)) / 2.0,
        0.0);
      let linear_arg_part = complex::new(
        0.0, p0_x * x as real + p0_y * y as real);
      let coherent_psi = ((square_arg_part + linear_arg_part) / hbar).exp();
      psi.set(x, y, coherent_psi);
    }
  }
  apply_boundary_conditions(&mut psi);

  // 2. Normalize the wavefunction to give the total probability of 1.
  normalize_wave_function(&mut psi);

  // 3. Compute baseline for probability -> color mapping.
  let render_options = RenderOpts {
    shades_num: 80,
  };

  // 4. Render animation.
  let palette = generate_palette(&render_options);
  let mut image = std::fs::File::create("target/qgolf.gif").unwrap();
  let mut encoder = gif::Encoder::new(
        &mut image, w as u16, h as u16, &palette).unwrap();
  use gif::SetParameter;
  encoder.set(gif::Repeat::Infinite).unwrap();
  
  for frame_index in 0 .. frames_count {
    println!("Rendering frame {} / {}..", frame_index + 1, &frames_count);
    let mut buffer = vec![0_u8; w * h];
    render_frame(&psi, &mut buffer, &render_options);
    
    let mut frame = gif::Frame::default();
    frame.width = w as u16;
    frame.height = h as u16;
    frame.buffer = std::borrow::Cow::Borrowed(&buffer);

    encoder.write_frame(&frame).unwrap();

    if individual_frames {
      let mut individual_image = std::fs::File::create(
          format!("target/qgolf_{:#04}.gif", frame_index + 1)).unwrap();;
      let mut individual_encoder = gif::Encoder::new(
          &mut individual_image, w as u16, h as u16, &palette).unwrap();
      use gif::SetParameter;
      encoder.set(gif::Repeat::Infinite).unwrap();
      individual_encoder.write_frame(&frame).unwrap();
    }

    evolve(&mut psi, &u, dt, hbar, steps_per_frame);
  }
}

