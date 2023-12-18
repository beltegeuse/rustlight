pub trait Poly {
    fn pdf(&self, x: f32) -> f32;
    fn cdf(&self, x: f32) -> f32;
    fn cdf_pn(&self, a: f32, b: f32, min_theta: f32, max_theta: f32) -> f32;
}
pub struct Poly4 {
    t: [f32; 5],
}
impl Poly4 {
    pub fn phase(g: f32) -> Self {
        let hterm = 1.0 + g * g;
        let hterm_sqrt = hterm.sqrt();

        let h_3_2 = 1.0 / (hterm * hterm_sqrt);
        let h_5_2 = 1.0 / (hterm.powi(2) * hterm_sqrt);
        let h_7_2 = 1.0 / (hterm.powi(3) * hterm_sqrt);
        let h_9_2 = 1.0 / (hterm.powi(4) * hterm_sqrt);
        let h_11_2 = 1.0 / (hterm.powi(5) * hterm_sqrt);

        // Taylor expension of  1.0/(1+g^2+2*g*Sin[theta])^(3/2)
        Poly4 {
            t: [
                h_3_2,                                                                    // c0
                -3.0 * h_5_2 * g,                                                         // c1
                (15.0 / 2.0) * h_7_2 * g.powi(2),                                         // c2
                0.5 * (g - 33.0 * g.powi(3) + g.powi(5)) * h_9_2,                         // c3
                -0.625 * (4.0 * g.powi(2) - 55.0 * g.powi(4) + 4.0 * g.powi(6)) * h_11_2, // c4
            ],
        }
    }

    pub fn tr_phase(
        pn: &crate::integrators::explicit::point_normal::EquiAngularSampling,
        sigma_t: f32,
        g: f32,
    ) -> Self {
        let hterm = 1.0 + g * g;
        let hterm_sqrt = hterm.sqrt();

        let h_3_2 = 1.0 / (hterm * hterm_sqrt);
        let h_5_2 = 1.0 / (hterm.powi(2) * hterm_sqrt);
        let h_7_2 = 1.0 / (hterm.powi(3) * hterm_sqrt);
        let h_9_2 = 1.0 / (hterm.powi(4) * hterm_sqrt);
        let h_11_2 = 1.0 / (hterm.powi(5) * hterm_sqrt);
        let d = pn.d_l;

        // Note that this constant get cancel out
        // when normalizating the PDF
        // let c0 = ((-delta - d) * sigma_t).exp();
        let c0 = 1.0;

        let power = |v: f32, i: i32| v.powi(i);
        Poly4 {
            t: [
                c0 * h_3_2,
                (c0 * (-1. * d * sigma_t + g * (-3. - 1. * d * g * sigma_t))) * h_5_2,
                (c0 * (7.5 * power(g, 2)
                    + 3. * d * g * (1.0 + power(g, 2)) * sigma_t
                    + 0.5 * d * power(1.0 + power(g, 2), 2) * sigma_t * (-1.0 + d * sigma_t)))
                    * h_7_2,
                (c0 * (0.5 * (g - 33.0 * power(g, 3) + power(g, 5))
                    - 7.5 * d * power(g, 2) * (1.0 + power(g, 2)) * sigma_t
                    - 1.5 * d * g * power(1.0 + power(g, 2), 2) * sigma_t * (-1.0 + d * sigma_t)
                    - 0.16666666666666666
                        * d
                        * power(1.0 + power(g, 2), 3)
                        * sigma_t
                        * (-2.0 + d * sigma_t)
                        * (-1.0 + d * sigma_t)))
                    * h_9_2,
                (c0 * (-2.5 * power(g, 2) + 34.375 * power(g, 4)
                    - 2.5 * power(g, 6)
                    - 0.5
                        * d
                        * (1. + power(g, 2))
                        * (g - 33. * power(g, 3) + power(g, 5))
                        * sigma_t
                    + 0.5
                        * d
                        * g
                        * power(1.0 + power(g, 2), 3)
                        * sigma_t
                        * (-2.0 + d * sigma_t)
                        * (-1.0 + d * sigma_t)
                    + 3.75 * d * power(g + power(g, 3), 2) * sigma_t * (-1. + d * sigma_t)
                    + 0.041666666666666664
                        * d
                        * power(1.0 + power(g, 2), 4)
                        * sigma_t
                        * (-5.0 + d * sigma_t * (11.0 + d * sigma_t * (-6.0 + d * sigma_t)))))
                    * h_11_2,
            ],
        }
    }

    pub fn tr(
        pn: &crate::integrators::explicit::point_normal::EquiAngularSampling,
        sigma_t: f32,
    ) -> Self {
        // let delta = pn.delta;
        let d_norm = pn.d_l;

        // Note that this constant get cancel out
        // when normalizating the PDF
        // let c0 = ((-delta - d_norm) * sigma_t).exp();
        let c0 = 1.0;

        Poly4 {
            t: [
                c0,
                -(c0 * d_norm * sigma_t),
                (c0 * d_norm * sigma_t * (-1.0 + d_norm * sigma_t)) / 2.0,
                -(c0 * d_norm * sigma_t * (-2.0 + d_norm * sigma_t) * (-1.0 + d_norm * sigma_t))
                    / 6.0,
                (c0 * d_norm
                    * sigma_t
                    * (-5.0
                        + d_norm
                            * sigma_t
                            * (11.0 + d_norm * sigma_t * (-6.0 + d_norm * sigma_t))))
                    / 24.0,
            ],
        }
    }
}
fn poly4(t: &[f32; 5], x: f32) -> f32 {
    t[0] + t[1] * x + t[2] * x * x + t[3] * x * x * x + t[4] * x * x * x * x
}
impl Poly for Poly4 {
    fn pdf(&self, x: f32) -> f32 {
        poly4(&self.t, x)
    }

    fn cdf(&self, x: f32) -> f32 {
        x * (self.t[0]
            + (self.t[1] * x) / 2.0
            + (self.t[2] * x * x) / 3.0
            + (self.t[3] * x * x * x) / 4.0
            + (self.t[4] * x * x * x * x) / 5.0)
    }

    fn cdf_pn(&self, a: f32, b: f32, min_theta: f32, max_theta: f32) -> f32 {
        let c1 = [
            -(b * self.t[0]) + a * self.t[1] - 6.0 * a * (self.t[3] - 20.0 * 0.0)
                + 2.0 * b * (self.t[2] - 12.0 * self.t[4] + 360.0 * 0.0),
            -(b * self.t[1]) + 2.0 * a * self.t[2] + 6.0 * b * (self.t[3] - 20.0 * 0.0)
                - 24.0 * a * (self.t[4] - 30.0 * 0.0),
            -(b * self.t[2]) + 3.0 * a * self.t[3] - 60.0 * a * 0.0
                + 12.0 * b * (self.t[4] - 30.0 * 0.0),
            -(b * self.t[3]) + 4.0 * a * self.t[4] + 20.0 * b * 0.0 - 120.0 * a * 0.0,
            -(b * self.t[4]) + 5.0 * a * 0.0 + 30.0 * b * 0.0,
        ];
        // Same as C1 but (a and b inversed) + some sign are also inverted
        // The sign inverted can be easily fixed by changing p3 and p4
        let c2 = [
            a * self.t[0] + b * self.t[1]
                - 6.0 * b * (self.t[3] - 20.0 * 0.0)
                - 2.0 * a * (self.t[2] - 12.0 * self.t[4] + 360.0 * 0.0),
            a * self.t[1] + 2.0 * b * self.t[2]
                - 6.0 * a * (self.t[3] - 20.0 * 0.0)
                - 24.0 * b * (self.t[4] - 30.0 * 0.0),
            a * self.t[2] + 3.0 * b * self.t[3]
                - 60.0 * b * 0.0
                - 12.0 * a * (self.t[4] - 30.0 * 0.0),
            a * self.t[3] + 4.0 * b * self.t[4] - 20.0 * a * 0.0 - 120.0 * b * 0.0,
            a * self.t[4] + 5.0 * b * 0.0 - 30.0 * a * 0.0,
        ];

        let p1 = poly4(&c1, max_theta);
        let p2 = poly4(&c1, min_theta);
        let p3 = poly4(&c2, max_theta);
        let p4 = poly4(&c2, min_theta);

        // Final expression
        p1 * max_theta.cos() - p2 * min_theta.cos() + p3 * max_theta.sin() - p4 * min_theta.sin()
    }
}
fn poly6(c: &[f32; 7], x: f32) -> f32 {
    c[0] + c[1] * x
        + c[2] * x * x
        + c[3] * x * x * x
        + c[4] * x * x * x * x
        + c[5] * x * x * x * x * x
        + c[6] * x * x * x * x * x * x
}
pub struct Poly6 {
    t: [f32; 7],
}
impl Poly6 {
    #[rustfmt::skip]
    pub fn phase(g: f32) -> Self {
        let hterm = 1.0 + g * g;
        let hterm_sqrt = hterm.sqrt();

        let h_3_2 = 1.0 / (hterm * hterm_sqrt);
        let h_5_2 = 1.0 / (hterm.powi(2) * hterm_sqrt);
        let h_7_2 = 1.0 / (hterm.powi(3) * hterm_sqrt);
        let h_9_2 = 1.0 / (hterm.powi(4) * hterm_sqrt);
        let h_11_2 = 1.0 / (hterm.powi(5) * hterm_sqrt);
        let h_13_2 = 1.0 / (hterm.powi(6) * hterm_sqrt);
        let h_15_2 = 1.0 / (hterm.powi(7) * hterm_sqrt);

        let power = |v: f32, i: i32| v.powi(i);
        Poly6 {
            t: [
                h_3_2,                                                                    // c0
                -3.0 * h_5_2 * g,                                                         // c1
                (15.0 / 2.0) * h_7_2 * g.powi(2),                                         // c2
                0.5 * (g - 33.0 * g.powi(3) + g.powi(5)) * h_9_2,                         // c3
                -0.625 * (4.0 * g.powi(2) - 55.0 * g.powi(4) + 4.0 * g.powi(6)) * h_11_2, // c4
                (-0.025*g+8.65*power(g,3)-69.275*power(g,5)+8.65*power(g,7)-0.025*power(g,9))*h_13_2,
                (power(g,2)*(0.3333333333333333-24.916666666666664*power(g,2)+137.1875*power(g,4)-24.916666666666664*power(g,6)+0.3333333333333333*power(g,8)))*h_15_2
            ]
        }
    }

    #[rustfmt::skip]
    pub fn tr_phase(
        pn: &crate::integrators::explicit::point_normal::EquiAngularSampling,
        sigma_t: f32,
        g: f32,
    ) -> Self {
        let hterm = 1.0 + g * g;
        let hterm_sqrt = hterm.sqrt();

        let h_3_2 = 1.0 / (hterm * hterm_sqrt);
        let h_5_2 = 1.0 / (hterm.powi(2) * hterm_sqrt);
        let h_7_2 = 1.0 / (hterm.powi(3) * hterm_sqrt);
        let h_9_2 = 1.0 / (hterm.powi(4) * hterm_sqrt);
        let h_11_2 = 1.0 / (hterm.powi(5) * hterm_sqrt);
        let h_13_2 = 1.0 / (hterm.powi(6) * hterm_sqrt);
        let h_15_2 = 1.0 / (hterm.powi(7) * hterm_sqrt);
        let d = pn.d_l;
        // Note that this constant get cancel out 
        // when normalizating the PDF
        //let c0 = ((-delta - d) * sigma_t).exp();
        let c0 = 1.0;

        let power = |v: f32, i: i32| v.powi(i);
        Poly6 {
            t: [
                c0 * h_3_2,
                (c0 * (-1. * d * sigma_t + g * (-3. - 1. * d * g * sigma_t))) * h_5_2,
                (c0 * (7.5 * power(g, 2)
                    + 3. * d * g * (1.0 + power(g, 2)) * sigma_t
                    + 0.5 * d * power(1.0 + power(g, 2), 2) * sigma_t * (-1.0 + d * sigma_t)))
                    * h_7_2,
                (c0 * (0.5 * (g - 33.0 * power(g, 3) + power(g, 5))
                    - 7.5 * d * power(g, 2) * (1.0 + power(g, 2)) * sigma_t
                    - 1.5 * d * g * power(1.0 + power(g, 2), 2) * sigma_t * (-1.0 + d * sigma_t)
                    - 0.16666666666666666
                        * d
                        * power(1.0 + power(g, 2), 3)
                        * sigma_t
                        * (-2.0 + d * sigma_t)
                        * (-1.0 + d * sigma_t)))
                    * h_9_2,
                (c0 * (-2.5 * power(g, 2) + 34.375 * power(g, 4)
                    - 2.5 * power(g, 6)
                    - 0.5
                        * d
                        * (1. + power(g, 2))
                        * (g - 33. * power(g, 3) + power(g, 5))
                        * sigma_t
                    + 0.5
                        * d
                        * g
                        * power(1.0 + power(g, 2), 3)
                        * sigma_t
                        * (-2.0 + d * sigma_t)
                        * (-1.0 + d * sigma_t)
                    + 3.75 * d * power(g + power(g, 3), 2) * sigma_t * (-1. + d * sigma_t)
                    + 0.041666666666666664
                        * d
                        * power(1.0 + power(g, 2), 4)
                        * sigma_t
                        * (-5.0 + d * sigma_t * (11.0 + d * sigma_t * (-6.0 + d * sigma_t)))))
                    * h_11_2,
                    (c0*(-0.025000000000000022*g-8.673617379884035e-18*power(g,2)+8.65000000000001*power(g,3)+1.27675647831893e-15*power(g,4)-69.27500000000006*power(g,5)-8.881784197001252e-16*power(g,6)+8.65000000000001*power(g,7)-0.025000000000000022*power(g,9)+d*(-0.13333333333333333+g*(0.375+g*(-0.6666666666666666+g*(10.25+g*(-40.708333333333336+g*(19.75+g*(-40.708333333333336+g*(10.25+g*(-0.6666666666666666+(0.375-0.13333333333333333*g)*g)))))))))*sigma_t+0.375*power(d,2)*(-2.1339198732229225+1.*g)*(-0.4686211570304514+1.*g)*power(1.+1.*power(g,2),2)*(10.861331446853594+g*(-0.3639501708796947+1.*g))*(0.0920697434649864+g*(-0.033508798866931463+1.*g))*power(sigma_t,2)-0.2916666666666665*power(d,3)*power(0.9999999999999991+1.*power(g,2),3)*(5.1987361271557475+g*(-2.156597464138244+1.*g))*(0.19235444453056028+g*(-0.41483110729032674+1.*g))*power(sigma_t,3)+0.08333333333333334*power(d,4)*power(1.+1.*power(g,2),4)*(0.9999999999999998+g*(-1.4999999999999996+1.*g))*power(sigma_t,4)-0.008333333333333331*power(d,5)*power(1.+1.*power(g,2),5)*power(sigma_t,5)))*h_13_2,
                    (c0*(power(g,2)*(0.3333333333333332+g*(-4.440892098500626e-16+g*(-24.24999999999999+g*(3.597122599785508e-14+g*(87.68749999999999+g*(8.748557434046234e-14+g*(224.54166666666666+g*(6.483702463810914e-14+g*(87.68749999999999+g*(1.332267629550188e-14+g*(-24.249999999999993+(-4.440892098500626e-16+0.33333333333333326*g)*g)))))))))))+d*(-0.08472222222222223+g*(0.25833333333333336+g*(-0.9902777777777778+g*(-1.1083333333333336+g*(-23.934722222222224+g*(77.46666666666668+g*(-88.18194444444445+g*(239.75000000000006+g*(-130.30555555555557+g*(239.75000000000006+g*(-88.18194444444445+g*(77.46666666666668+g*(-23.934722222222224+g*(-1.1083333333333336+g*(-0.9902777777777778+(0.25833333333333336-0.08472222222222223*g)*g)))))))))))))))*sigma_t+power(d,2)*(0.29305555555555557+g*(-0.875+g*(4.531944444444445+g*(-14.875+g*(41.018055555555556+g*(-62.12500000000001+g*(127.97361111111111+g*(-118.125+g*(182.38888888888889+g*(-118.125+g*(127.97361111111111+g*(-62.12500000000001+g*(41.018055555555556+g*(-14.875+g*(4.531944444444445+(-0.875+0.29305555555555557*g)*g)))))))))))))))*power(sigma_t,2)+power(d,3)*(-0.2916666666666667+g*(0.7916666666666667+g*(-4.208333333333334+g*(8.458333333333334+g*(-19.416666666666668+g*(31.208333333333336+g*(-44.458333333333336+g*(56.87500000000001+g*(-57.91666666666667+g*(56.87500000000001+g*(-44.458333333333336+g*(31.208333333333336+g*(-19.416666666666668+g*(8.458333333333334+g*(-4.208333333333334+(0.7916666666666667-0.2916666666666667*g)*g)))))))))))))))*power(sigma_t,3)+power(d,4)*(0.11805555555555557+g*(-0.25+g*(1.2569444444444444+g*(-1.75+g*(5.180555555555555+g*(-5.25+g*(11.29861111111111+g*(-8.75+g*(14.51388888888889+g*(-8.75+g*(11.29861111111111+g*(-5.25+g*(5.180555555555555+g*(-1.75+g*(1.2569444444444444+(-0.25+0.11805555555555557*g)*g)))))))))))))))*power(sigma_t,4)+power(d,5)*(-0.020833333333333336+g*(0.025+g*(-0.16666666666666669+g*(0.175+g*(-0.5833333333333334+g*(0.525+g*(-1.1666666666666667+g*(0.875+g*(-1.4583333333333335+g*(0.875+g*(-1.1666666666666667+g*(0.525+g*(-0.5833333333333334+g*(0.175+g*(-0.16666666666666669+(0.025-0.020833333333333336*g)*g)))))))))))))))*power(sigma_t,5)+power(d,6)*(0.001388888888888889+0.011111111111111112*power(g,2)+0.03888888888888889*power(g,4)+0.07777777777777778*power(g,6)+0.09722222222222222*power(g,8)+0.07777777777777778*power(g,10)+0.03888888888888889*power(g,12)+0.011111111111111112*power(g,14)+0.001388888888888889*power(g,16))*power(sigma_t,6)))*h_15_2
            ],
        }
    }

    pub fn tr(
        equiangular: &crate::integrators::explicit::point_normal::EquiAngularSampling,
        sigma_t: f32,
    ) -> Self {
        let d_norm = equiangular.d_l;

        // Note that this constant get cancel out
        // when normalizating the PDF
        // let c0 = ((-delta - d_norm) * sigma_t).exp();
        let c0 = 1.0;

        Self {
            t: [
                c0,
                -(c0 * d_norm * sigma_t),
                (c0 * d_norm * sigma_t * (-1.0 + d_norm * sigma_t)) / 2.0,
                -(c0 * d_norm * sigma_t * (-2.0 + d_norm * sigma_t) * (-1.0 + d_norm * sigma_t))
                    / 6.0,
                (c0 * d_norm
                    * sigma_t
                    * (-5.0
                        + d_norm
                            * sigma_t
                            * (11.0 + d_norm * sigma_t * (-6.0 + d_norm * sigma_t))))
                    / 24.0,
                -(c0 * d_norm
                    * sigma_t
                    * (16.0
                        + d_norm
                            * sigma_t
                            * (-45.0
                                + d_norm
                                    * sigma_t
                                    * (35.0 + d_norm * sigma_t * (-10.0 + d_norm * sigma_t)))))
                    / 120.0,
                (c0 * d_norm
                    * sigma_t
                    * (-61.0
                        + d_norm
                            * sigma_t
                            * (211.0
                                + d_norm
                                    * sigma_t
                                    * (-210.0
                                        + d_norm
                                            * sigma_t
                                            * (85.0
                                                + d_norm
                                                    * sigma_t
                                                    * (-15.0 + d_norm * sigma_t))))))
                    / 720.0,
            ],
        }
    }
}
impl Poly for Poly6 {
    fn pdf(&self, x: f32) -> f32 {
        poly6(&self.t, x)
    }

    fn cdf(&self, x: f32) -> f32 {
        x * (self.t[0]
            + (self.t[1] * x) / 2.0
            + (self.t[2] * x * x) / 3.0
            + (self.t[3] * x * x * x) / 4.0
            + (self.t[4] * x * x * x * x) / 5.0
            + (self.t[5] * x * x * x * x * x) / 6.0
            + (self.t[6] * x * x * x * x * x * x) / 7.0)
    }

    fn cdf_pn(&self, a: f32, b: f32, min_theta: f32, max_theta: f32) -> f32 {
        let t = &self.t;
        let c1 = [
            -(b * t[0]) + a * t[1] - 6.0 * a * (t[3] - 20.0 * t[5])
                + 2.0 * b * (t[2] - 12.0 * t[4] + 360.0 * t[6]),
            -(b * t[1]) + 2.0 * a * t[2] + 6.0 * b * (t[3] - 20.0 * t[5])
                - 24.0 * a * (t[4] - 30.0 * t[6]),
            -(b * t[2]) + 3.0 * a * t[3] - 60.0 * a * t[5] + 12.0 * b * (t[4] - 30.0 * t[6]),
            -(b * t[3]) + 4.0 * a * t[4] + 20.0 * b * t[5] - 120.0 * a * t[6],
            -(b * t[4]) + 5.0 * a * t[5] + 30.0 * b * t[6],
            -(b * t[5]) + 6.0 * a * t[6],
            -(b * t[6]),
        ];
        // Same as C1 but (a and b inversed) + some sign are also inverted
        // The sign inverted can be easily fixed by changing p3 and p4
        let c2 = [
            a * t[0] + b * t[1]
                - 6.0 * b * (t[3] - 20.0 * t[5])
                - 2.0 * a * (t[2] - 12.0 * t[4] + 360.0 * t[6]),
            a * t[1] + 2.0 * b * t[2]
                - 6.0 * a * (t[3] - 20.0 * t[5])
                - 24.0 * b * (t[4] - 30.0 * t[6]),
            a * t[2] + 3.0 * b * t[3] - 60.0 * b * t[5] - 12.0 * a * (t[4] - 30.0 * t[6]),
            a * t[3] + 4.0 * b * t[4] - 20.0 * a * t[5] - 120.0 * b * t[6],
            a * t[4] + 5.0 * b * t[5] - 30.0 * a * t[6],
            a * t[5] + 6.0 * b * t[6],
            a * t[6],
        ];

        // Evaluate polynomes
        let p1 = poly6(&c1, max_theta);
        let p2 = poly6(&c1, min_theta);
        let p3 = poly6(&c2, max_theta);
        let p4 = poly6(&c2, min_theta);

        // Final expression
        p1 * max_theta.cos() - p2 * min_theta.cos() + p3 * max_theta.sin() - p4 * min_theta.sin()
    }
}

// Implement agnostic clamping
// Can be helpful to see the numerical inversion
// with the clamping
pub struct PolyClamped<T: Poly> {
    pub poly: T,
    pub clamp_angle: f32,
    pub clamp_pdf: f32,
    pub clamp_cdf: f32,
}
impl<T: Poly> PolyClamped<T> {
    pub fn new(poly: T, range: (f32, f32), clamp_angle: f32) -> Self {
        if clamp_angle >= range.1 {
            Self {
                poly,
                clamp_angle: std::f32::MAX,
                clamp_pdf: 0.0,
                clamp_cdf: 0.0,
            }
        } else {
            // Evaluate the polynom on two domains
            let clamp_pdf = poly.pdf(clamp_angle);
            let clamp_cdf = poly.cdf(clamp_angle);
            Self {
                poly,
                clamp_angle,
                clamp_pdf,
                clamp_cdf,
            }
        }
    }
}

impl<T: Poly> Poly for PolyClamped<T> {
    fn pdf(&self, x: f32) -> f32 {
        // Give back the
        if x < self.clamp_angle {
            self.poly.pdf(x)
        } else {
            self.clamp_pdf
        }
    }

    fn cdf(&self, x: f32) -> f32 {
        if x < self.clamp_angle {
            self.poly.cdf(x)
        } else {
            self.clamp_cdf + (x - self.clamp_angle) * self.clamp_pdf
        }
    }

    fn cdf_pn(&self, _a: f32, _b: f32, _min_theta: f32, _max_theta: f32) -> f32 {
        unimplemented!(); // Not implemented for the moment
                          // This requires to compute the pdf and the CDF in other way
                          // One potential implementation is to provide another struct: PolyClampedPN
    }
}
